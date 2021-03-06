function convertInf!(A::VecOrMat{Float64},infbnd=1.1e20)
    infs = isinf.(A)
    A[infs] = sign.(A[infs])*infbnd
    return nothing
end

function get_jacobian_values(jac::Union{Vector,Matrix,SparseMatrixCSC},inds::Vector{Int})
    if ndims(jac) == 1
        return jac
    else
        return jac[inds]
    end
end

function createProblem(fun, x0, lb, ub; iI=Int[], jI=Int[], iE=Int[], jE=Int[])
    # call function
    res = fun(x0)
    if length(res) == 3  # inequality only
        J, c, fail = res
        ceq = Float64[]
        gradprovided = false
    elseif length(res) == 4
        J, c, ceq, fail = res
        gradprovided = false
    elseif length(res) == 5  # inequality only
        J, c, gJ, gc, fail = res
        ceq = Float64[]
        gceq = Float64[]
        gradprovided = true
    else
        J, c, ceq, gJ, gc, gceq, fail = res
        gradprovided = true
    end

    # States
    n = length(x0)
    @assert length(lb) == n
    @assert length(ub) == n

    # Constraints
    mI = length(c)
    mE = length(ceq)
    m = mI + mE

    # Snopt vars
    nF = 1 + m  # 1 objective + constraints
    ObjRow = 1  # objective is first thing returned, then constraints

    # Gradients
    @assert length(iI) == length(jI)
    @assert length(iE) == length(jE)
    dense_jacobian = true
    if gradprovided
        # Check if sparsity structure was passed in
        if isempty(c)
            c_structure_provided = true
        else
            if isempty(iI) || isempty(jI)
                c_structure_provided = false
            else
                c_structure_provided = true
            end
        end
        if isempty(ceq)
            ceq_structure_provided = true
        else
            if isempty(iE) || isempty(jE)
                ceq_structure_provided = false
            else
                ceq_structure_provided = true
            end
        end

        # Determine if the jacobians are sparse
        if c_structure_provided && ceq_structure_provided # Must specify sparsity structure
            if issparse(gc) || issparse(gceq)  # Returns sparse array
                dense_jacobian = false
            elseif ndims(gc) == 1 || ndims(gceq) == 1  # Returns vector
                dense_jacobian = false
            end
        end

    end
    if dense_jacobian
        # Assume Dense Jacobian if not provided
        lenG = nF*n
        iGfun = Array{Int32}(lenG)
        jGvar = Array{Int32}(lenG)
        iGfun[1:n] = 1
        jGvar[1:n] = 1:n
        k = n+1
        for j = 1:n
            for i = 2:nF
                iGfun[k] = i
                jGvar[k] = j
                k += 1
            end
        end

        # Generate linear indices
        lenG_c = n*mI
        lenG_ceq = n*mE
        c_inds = collect(1:lenG_c)
        ceq_inds = collect(1:lenG_ceq)
    else
        iJ,jJ = ones(n), 1:n  # Assume objective gradient is dense

        # Convert to indices
        c_inds = sub2ind((mI,n),iI,jI)
        ceq_inds = sub2ind((mE,n),iE,jE)
        lenG_c = length(c_inds)
        lenG_ceq = length(ceq_inds)

        # Combine row,col sets to get structure for combined jacobian
        iG = [iJ; 1+iI; 1+lenG_c+iE] # Shift rows when stacking vertically
        jG = [jJ; jI; jE]
        iGfun = convert.(Int32,iG)
        jGvar = convert.(Int32,jG)
        lenG = length(iG)

    end

    # bound constriaints
    xlow = lb
    xupp = ub
    Flow = -Inf*ones(nF)
    Fupp = zeros(nF)  # TODO: currently c <= 0, but perhaps change
    if !isempty(ceq) # equality constraints
        Flow[nF - length(ceq) + 1 : nF] = 0.0
    end
    convertInf!(xlow)
    convertInf!(xupp)
    convertInf!(Flow)

    # Pre-allocate arrays
    F = zeros(nF)
    if gradprovided
        G = zeros(lenG)
    else
        G = Float64[]
    end

    # Auto-gen the user defined function
    function usrfun(x)
        res = fun(x)
        if length(res) == 3
            J, c, fail = res
            ceq = Float64[]
            gradprovided = false
        elseif length(res) == 4
            J, c, ceq, fail = res
            gradprovided = false
        elseif length(res) == 5
            J, c, gJ, gc, fail = res
            ceq = Float64[]
            gceq = Float64[]
            gradprovided = true
        else
            J, c, ceq, gJ, gc, gceq, fail = res
            gradprovided = true
        end

        # Function values
        F[1] = J
        F[1+(1:mI)] = c
        F[1+mI+(1:mE)] = ceq

        if gradprovided
            # Cost gradient
            G[1:n] = gJ

            # Constraint Jacobians
            jac_c = get_jacobian_values(gc,c_inds)
            jac_ceq = get_jacobian_values(gceq,ceq_inds)

            G[n+(1:lenG_c)] = jac_c
            if lenG_ceq > 0
                G[n+lenG_c+(1:lenG_ceq)] = jac_ceq
            end
        end
        return F,G,fail
    end # usrfun

    return SnoptProblem(n,m,iGfun,jGvar,xlow,xupp,Flow,Fupp,usrfun)
end

mutable struct SnoptProblem
    x::Vector{Float64}

    n::Int  # Num vars
    m::Int  # Num cons

    nF::Int
    lenG::Int
    neG::Int
    objRow::Int

    iGfun::Vector{Int32}
    jGvar::Vector{Int32}

    xlow::Vector{Float64}
    xupp::Vector{Float64}
    Flow::Vector{Float64}
    Fupp::Vector{Float64}

    # Callbacks
    usrfun::Function  # F(x) form for Snopt

    function SnoptProblem(n,m,iGfun,jGvar,xlow,xupp,Flow,Fupp,usrfun,objRow=1)
        x0 = zeros(n)
        nF = 1 + m  # 1 objective + constraints
        lenG = length(iGfun)
        neG = lenG
        @assert length(jGvar) == lenG

        new(x0,n,m,nF,lenG,neG,objRow,iGfun,jGvar,xlow,xupp,Flow,Fupp,usrfun)
    end
end
