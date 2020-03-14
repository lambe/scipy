from numpy import sqrt, inner, zeros, inf, finfo, abs
from numpy.linalg import norm

from .utils import make_system

__all__ = ['symmlq']


def symmlq(A, b, x0=None, shift=0.0, tol=1e-5, maxiter=None,
           M=None, callback=None, show=False, check=False):
    """
    Use Symmetric LQ iteration to solve Ax=b.

    SYMMLQ solves A*x = b for real symmetric matrix A. Unlike
    the Conjugate Gradient method, A can be indefinite.

    If shift != 0 then the method solves (A - shift*I)x = b

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
        The real symmetric N-by-N matrix of the linear system
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).

    Returns
    -------
    x : {array, matrix}
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : illegal input or breakdown

    Other Parameters
    ----------------
    x0  : {array, matrix}
        Starting guess for the solution.
    tol : float
        Tolerance to achieve. The algorithm terminates when the relative
        residual is below `tol`.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, dense matrix, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.

    References
    ----------
    Solution of sparse indefinite systems of linear equations,
        C. C. Paige and M. A. Saunders (1975),
        SIAM J. Numer. Anal. 12(4), pp. 617-629.
        https://web.stanford.edu/group/SOL/software/symmlq/

    This file is a translation of the following MATLAB implementation:
        https://web.stanford.edu/group/SOL/software/symmlq/symmlq-matlab.zip

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    matvec = A.matvec
    psolve = M.matvec

    first = 'Enter SYMMLQ.   '
    last = 'Exit  SYMMLQ.   '

    n = A.shape[0]

    if maxiter is None:
        maxiter = 5 * n

    msg = {-1: ' beta2 = 0.  If M = I, b and x are eigenvectors',
           0: ' beta1 = 0.  The exact solution is  x = 0',
           1: ' Requested accuracy achieved, as determined by rtol',
           2: ' Reasonable accuracy achieved, given eps',
           3: ' x has converged to an eigenvector',
           4: ' acond has exceeded 0.1/eps',
           5: ' The iteration limit was reached',
           6: ' aprod  does not define a symmetric matrix',
           7: ' msolve does not define a symmetric matrix',
           8: ' msolve does not define a pos-def preconditioner'
           }

    if show:
        print(first + 'Solution of symmetric Ax = b')
        print(first + 'n      =  %3g     shift  =  %23.14e' % (n, shift))
        print(first + 'itnlim =  %3g     rtol   =  %11.2e' % (maxiter, tol))
        print()

    istop = 0
    itn = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0

    xtype = x.dtype

    eps = finfo(xtype).eps

    r1 = b - A*x
    y = psolve(r1)

    beta1 = inner(r1, y)
    if beta1 < 0:
        # Error code 8
        raise ValueError('indefinite preconditioner')
    elif beta1 == 0:
        return (postprocess(x), 0)
    else:
        beta1 = sqrt(beta1)

    b1 = y[0]
    s = 1.0 / beta1
    v = s * y

    if check:
        # see if A is symmetric
        w = matvec(y)
        r2 = matvec(w)
        s = inner(w, w)
        t = inner(y, r2)
        z = abs(s - t)
        epsa = (s + eps) * eps**(1.0 / 3.0)
        if z > epsa:
            # Error code 6
            raise ValueError('non-symmetric matrix')

        # see if M is symmetric
        r2 = psolve(y)
        s = inner(y,y)
        t = inner(r1, r2)
        z = abs(s - t)
        epsa = (s + eps) * eps**(1.0/3.0)
        if z > epsa:
            # Error code 7
            raise ValueError('non-symmetric preconditioner')

    # TODO: carry on writing solver
    # Set up y for the second Lanczos vector.
    # Again, y is beta * P * v2  where  P = C^(-1).
    # y and beta will be zero or very small if Abar = I or constant * I.

    y -= shift * v
    alfa = inner(v, y)
    y -= (alfa / beta1) * r1

    # Make sure  r2  will be orthogonal to the first  v.

    z = inner(v, y)
    s = inner(v, v)
    y -= (z / s) * v
    r2 = y.copy()

    y = psolve(r2)
    oldb = beta1
    beta = inner(r2, y)
    if beta < 0:
        # istop = 8
        raise ValueError('indefinite preconditioner')

    #  Cause termination (later) if beta is essentially zero.

    beta = sqrt(beta)
    if beta <= eps:
        # istop = -1
        raise ValueError("beta2 = 0.  If M = I, b and x are eigenvectors")

    #  See if the local reorthogonalization achieved anything.

    denom = sqrt(s) * norm(r2) + eps
    s = z / denom
    t = inner(v, r2)
    t = t / denom

    if show:
        print('beta1 =  %10.2e   alpha1 =  %9.2e' % (beta1, alfa))
        print('(v1, v2) before and after  %14.2e' % s)
        print('local reorthogonalization  %14.2e' % t)

    #  Initialize other quantities.
    cgnorm = beta1
    rhs2 = 0
    tnorm = alfa ** 2 + beta ** 2
    gbar = alfa
    bstep = 0
    ynorm2 = 0
    dbar = beta
    snprod = 1
    gmax = abs(alfa) + eps
    rhs1 = beta1
    x1cg = 0
    gmin = gmax
    qrnorm = beta1

    head1 = '   Itn     x(1)(cg)  normr(cg)  r(minres)'
    head2 = '    bstep    anorm    acond'
    if show:
        print(head1 + head2)

    str1 = '%6g %12.5e %10.3e' % (itn, x1cg, cgnorm)
    str2 = ' %10.3e  %8.1e' % (qrnorm, bstep / beta1)
    if show:
        print(str1 + str2)

    # ------------------------------------------------------------------
    # Main iteration loop.
    # ------------------------------------------------------------------
    # Estimate various norms and test for convergence.

    while itn < maxiter:
        itn += 1
        anorm = sqrt(tnorm)
        ynorm = sqrt(ynorm2)
        epsa = anorm * eps
        epsx = anorm * ynorm * eps
        epsr = anorm * ynorm * tol
        diag = gbar

        if diag == 0: diag = epsa

        lqnorm = sqrt(rhs1 ** 2 + rhs2 ** 2)
        qrnorm = snprod * beta1
        cgnorm = qrnorm * beta / abs(diag)

        # Estimate  Cond(A).
        # In this version we look at the diagonals of  L  in the
        # factorization of the tridiagonal matrix,  T = L*Q.
        # Sometimes, T(k) can be misleadingly ill-conditioned when
        # T(k+1) is not, so we must be careful not to overestimate acond

        if lqnorm < cgnorm:
            acond = gmax / gmin
        else:
            denom = min(gmin, abs(diag))
            acond = gmax / denom

        zbar = rhs1 / diag
        z = (snprod * zbar + bstep) / beta1
        x1lq = x[0] + b1 * bstep / beta1
        x1cg = x[0] + w[0] * zbar + b1 * z

        # See if any of the stopping criteria are satisfied.
        # In rare cases, istop is already -1 from above
        # (Abar = const * I).

        if istop == 0:
            if itn >= maxiter:
                istop = 5
            if acond >= 0.1 / eps:
                istop = 4
            if epsx >= beta1:
                istop = 3
            if cgnorm <= epsx:
                istop = 2
            if cgnorm <= epsr:
                istop = 1

        prnt = False
        if n <= 40:
            prnt = True
        if itn <= 20:
            prnt = True
        if itn >= maxiter - 10:
            prnt = True
        if itn % 10 == 0:
            prnt = True
        if cgnorm <= 10.0 * epsx:
            prnt = True
        if cgnorm <= 10.0 * epsr:
            prnt = True
        if acond >= 0.01 / eps:
            prnt = True
        if istop != 0:
            prnt = True

        if prnt and show:
            str1 = '%6g %12.5e %10.3e' % (itn, x1cg, cgnorm)
            str2 = ' %10.3e  %8.1e' % (qrnorm, bstep / beta1)
            str3 = ' %8.1e %8.1e' % (anorm, acond)
            print(str1 + str2 + str3)

        if istop != 0:
            break

        # Obtain the current Lanczos vector  v = (1 / beta)*y
        # and set up  y  for the next iteration.

        s = 1 / beta
        v = s * y
        y = matvec(v)
        itn += 1
        y -= shift * v
        y -= (beta / oldb) * r1
        alfa = inner(v, y)
        y -= (alfa / beta) * r2
        r1 = r2.copy()
        r2 = y.copy()
        y = psolve(r2)
        oldb = beta
        beta = inner(r2, y)

        if beta < 0:
            istop = 6
            break

        beta = sqrt(beta)
        tnorm = tnorm + alfa ** 2 + oldb ** 2 + beta ** 2

        # Compute the next plane rotation for Q.

        gamma = sqrt(gbar ** 2 + oldb ** 2)
        cs = gbar / gamma
        sn = oldb / gamma
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -  cs * beta

        # Update  X.

        z = rhs1 / gamma
        s = z * cs
        t = z * sn
        x += s * w + t * v
        w *= sn
        w -= cs * v

        if callback is not None:
            callback(x)

        # Accumulate the step along the direction b, and go round again.

        bstep = snprod * cs * z + bstep
        snprod = snprod * sn
        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        ynorm2 = z ** 2 + ynorm2
        rhs1 = rhs2 - delta * z
        rhs2 = -  epsln * z
    # end while

    # ------------------------------------------------------------------
    # End of main iteration loop.
    # ------------------------------------------------------------------

    # Move to the CG point if it seems better.
    # In this version of SYMMLQ, the convergence tests involve
    # only cgnorm, so we're unlikely to stop at an LQ point,
    # EXCEPT if the iteration limit interferes.

    if cgnorm < lqnorm:
        zbar = rhs1 / diag
        bstep = snprod * zbar + bstep
        ynorm = sqrt(ynorm2 + zbar ** 2)
        x += zbar * w

    # Add the step along b.

    bstep = bstep / beta1
    y = psolve(b)
    x += bstep * y

    # Compute the final residual,  r1 = b - (A - shift*I)*x.

    y = matvec(x)
    itn += 1
    y -= shift * x
    r1 = b - y
    rnorm = norm(r1)
    xnorm = norm(x)

    # ==================================================================
    # Display final status.
    # ==================================================================

    if show:
        fmt = ' istop   =  %3g               itn   =   %5g'
        print(last + fmt % (istop, itn))
        fmt = ' anorm   =  %12.4e      acond =  %12.4e'
        print(last + fmt % (anorm, acond))
        fmt = ' rnorm   =  %12.4e      xnorm =  %12.4e'
        print(last + fmt % (rnorm, xnorm))
        print(last + msg[istop])

    if istop == 6:
        info = maxiter
    else:
        info = 0

    return (postprocess(x), info)


if __name__ == '__main__':
    from numpy import arange
    from scipy.sparse import spdiags

    n = 10

    residuals = []

    def cb(x):
        residuals.append(norm(b - A*x))

    # A = poisson((10,),format='csr')
    A = spdiags([arange(1, n+1, dtype=float)], [0], n, n, format='csr')
    M = spdiags([1.0/arange(1, n+1, dtype=float)], [0], n, n, format='csr')
    A.psolve = M.matvec
    b = zeros(A.shape[0])
    x = symmlq(A, b, tol=1e-12, maxiter=None, callback=cb)
