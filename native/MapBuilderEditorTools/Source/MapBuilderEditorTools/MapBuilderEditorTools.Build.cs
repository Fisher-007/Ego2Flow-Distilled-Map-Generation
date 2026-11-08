using UnrealBuildTool;
public class MapBuilderEditorTools : ModuleRules
{
    public MapBuilderEditorTools(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine" });
        PrivateDependencyModuleNames.AddRange(new string[] { "AssetTools", "MeshUtilities", "UnrealEd", "MeshDescription", "StaticMeshDescription" });
    }
}
