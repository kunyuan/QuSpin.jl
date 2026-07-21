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
        has_documentation = if isdefined(Docs, :hasdoc)
            getfield(Docs, :hasdoc)(QuSpin, name)
        else
            Docs.doc(Docs.Binding(QuSpin, name)) !== nothing
        end
        @test has_documentation
    end
end
