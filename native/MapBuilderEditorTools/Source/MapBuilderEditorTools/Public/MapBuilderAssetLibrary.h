#pragma once
#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "MapBuilderAssetLibrary.generated.h"

class UStaticMesh;
class UMeshComponent;

/** Expose Unreal's existing reference-preserving package copy to build scripts. */
UCLASS()
class MAPBUILDEREDITORTOOLS_API UMapBuilderAssetLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:
    UFUNCTION(BlueprintCallable, Category="MAP|Editor")
    static bool CopyAssetPackages(const TMap<FString, FString>& SourceToDestination);

    /** Repeat source geometry in a distinct asset, preserving native material slots. */
    UFUNCTION(BlueprintCallable, Category="MAP|Editor")
    static bool RepeatStaticMesh(UStaticMesh* Source, UStaticMesh* Destination, const TArray<FTransform>& Copies);
    /** Reuse Unreal mesh conversion to preserve the current native component geometry. */
    UFUNCTION(BlueprintCallable, Category="MAP|Editor")
    static UStaticMesh* BakeMeshComponents(const TArray<UMeshComponent*>& Components, const FTransform& RootTransform, const FString& PackageName);
    /** Copy native simple collision into a distinct edited mesh and rebuild physics. */
    UFUNCTION(BlueprintCallable, Category="MAP|Editor")
    static bool CopySimpleCollision(UStaticMesh* Source, UStaticMesh* Destination);
};
