@testset "exported public API has docstrings" begin
    exported = sort!(
        filter(
            name -> Base.isexported(QuSpin, name),
            collect(names(QuSpin; all=false, imported=false)),
        );
        by=string,
    )
    @test length(exported) == 131
    for name in exported
        @test Docs.hasdoc(QuSpin, name)
    end
end
