using Test
include("ExprManipulation.jl")

"""
This macro is a shorthand for the summation over site pairs,
taking into account the right multiplicity.

This macro allows to write

```
@sitesum ss ki kj k expr(ki,kj,k, ...)
```

and obtain the sum

```
sum( m * expr(ki,kj,k) for (ki,kj,k,m) in zip(ss.ki,ss.kj,ss.xk,ss.m))
```

Arguing that the first expression
is a little more akin to the mathematical notation
than the second one.

"""
macro sitesum(siteSum,ki,kj,xk,expr)

    # One needs to escape expr with esc
    # otherwise all symbols in it are seen
    # as belonging to the module where this macro is defined.
    # But this then requires escaping also ki kj and k
    # in the for () part.
    # This would not nice because this means we have ABSOLUTELY
    # no macro hygiene.
    # So we have to replace ki kj and k
    # with safe symbols in the original expression first.
    newki = gensym(ki)
    newkj = gensym(kj)
    newxk = gensym(xk)

    expr = ExprManipulation.replace(expr,ki,newki)
    expr = ExprManipulation.replace(expr,kj,newkj)
    expr = ExprManipulation.replace(expr,xk,newxk)

    quote
        ki_ = $(esc(siteSum)).ki
        kj_ = $(esc(siteSum)).kj
        xk_ = $(esc(siteSum)).xk
        m_ = $(esc(siteSum)).m
        sum(m * $(esc(expr))
            for ($(esc(newki)),$(esc(newkj)), $(esc(newxk)), m)
                in zip(ki_,kj_,xk_,m_))
    end
end



## Test


function test()
    func(a,b,c,d,e) = a*b*c+d+e

    expected_values(ss,f,d,e) = sum(
        m * f(ki,kj,k,d,e)
        for (ki,kj,k,m) in zip(ss.ki,ss.kj,ss.xk,ss.m)
)
    @testset begin
        ss = (ki=[1],
              kj=[2],
              xk=[3],
              m=[4])

        println(@macroexpand (@sitesum ss a b c func(a,b,c,1,1)))
        @test  (@sitesum ss a b c func(a,b,c,1,1)) == expected_values(ss,func,1,1)

        ss = (ki=[1,1],
              kj=[2,3],
              xk=[3,4],
              m=[4,5])

        @test  (@sitesum ss a b c func(a,b,c,1,1)) ==  expected_values(ss,func,1,1)
    end




end

