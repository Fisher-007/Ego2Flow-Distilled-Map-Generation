#include "MapBuilderAssetLibrary.h"
#include "AssetToolsModule.h"
#include "IAssetTools.h"
#include "Modules/ModuleManager.h"
#include "Engine/StaticMesh.h"
#include "StaticMeshAttributes.h"
#include "StaticMeshOperations.h"

bool UMapBuilderAssetLibrary::CopyAssetPackages(const TMap<FString, FString>& SourceToDestination)
{
    return FModuleManager::LoadModuleChecked<FAssetToolsModule>("AssetTools")
        .Get().AdvancedCopyPackages(SourceToDestination, true, true);
}

bool UMapBuilderAssetLibrary::RepeatStaticMesh(UStaticMesh* Source, UStaticMesh* Destination, const TArray<FTransform>& Copies)
{
    if (!Source || !Destination || Source == Destination || Copies.Num() == 0)
    {
        return false;
    }
    const FMeshDescription* Original = Source->GetMeshDescription(0);
    if (!Original)
    {
        return false;
    }
    FMeshDescription Repeated;
    FStaticMeshAttributes Attributes(Repeated);
    Attributes.Register();
    for (const FTransform& Transform : Copies)
    {
        FStaticMeshOperations::FAppendSettings Settings;
        Settings.MeshTransform = Transform;
        for (int32 Channel = 0; Channel < FStaticMeshOperations::FAppendSettings::MAX_NUM_UV_CHANNELS; ++Channel)
        {
            Settings.bMergeUVChannels[Channel] = true;
        }
        FStaticMeshOperations::AppendMeshDescription(*Original, Repeated, Settings);
    }
    const int32 ExpectedVertices = Original->Vertices().Num() * Copies.Num();
    if (Repeated.Vertices().Num() != ExpectedVertices)
    {
        return false;
    }
    Destination->Modify();
    Destination->SetNumSourceModels(1);
    Destination->GetSourceModel(0).BuildSettings = Source->GetSourceModel(0).BuildSettings;
    Destination->StaticMaterials = Source->StaticMaterials;
    Destination->CreateMeshDescription(0, MoveTemp(Repeated));
    Destination->CommitMeshDescription(0);
    Destination->Build(false);
    Destination->PostEditChange();
    Destination->MarkPackageDirty();
    UE_LOG(LogTemp, Display, TEXT("MAP repeated mesh %s: %d copies, %d vertices, %d material slots"),
        *Destination->GetPathName(), Copies.Num(), ExpectedVertices, Destination->StaticMaterials.Num());
    return true;
}

#include "MeshUtilities.h"
#include "Components/MeshComponent.h"

UStaticMesh* UMapBuilderAssetLibrary::BakeMeshComponents(const TArray<UMeshComponent*>& Components, const FTransform& RootTransform, const FString& PackageName)
{
    return FModuleManager::LoadModuleChecked<IMeshUtilities>("MeshUtilities")
        .ConvertMeshesToStaticMesh(Components, RootTransform, PackageName);
}

#include "PhysicsEngine/BodySetup.h"

bool UMapBuilderAssetLibrary::CopySimpleCollision(UStaticMesh* Source, UStaticMesh* Destination)
{
    if (!Source || !Destination || Source == Destination || !Source->BodySetup)
    {
        return false;
    }
    Destination->Modify();
    Destination->CreateBodySetup();
    UBodySetup* Body = Destination->BodySetup;
    Body->Modify();
    Body->CopyBodyPropertiesFrom(Source->BodySetup);
    Body->CollisionTraceFlag = Source->BodySetup->CollisionTraceFlag;
    Body->InvalidatePhysicsData();
    Body->CreatePhysicsMeshes();
    Destination->PostEditChange();
    Destination->MarkPackageDirty();
    return true;
}
