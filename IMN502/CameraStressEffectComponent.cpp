#include "CameraStressEffectComponent.h"
#include "Kismet/GameplayStatics.h"

UCameraStressEffectComponent::UCameraStressEffectComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    StressLevel = 0.0f;
    CurrentStressLevel = 0.0f;
}

void UCameraStressEffectComponent::BeginPlay()
{
    Super::BeginPlay();
    Camera = Cast<UCameraComponent>(GetOwner()->GetComponentByClass(UCameraComponent::StaticClass()));
}

void UCameraStressEffectComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (bStressActive)
    {
        float Time = GetWorld()->GetTimeSeconds() - StressStartTime; // StressStartTime à définir dans TriggerStressEffect
        CurrentStressLevel = StressCurve ? 
			FMath::Clamp(StressCurve->GetFloatValue(Time) * StressLevel, 0.f, 1.f) :
			FMath::FInterpTo(CurrentStressLevel, StressLevel, DeltaTime, 2.5f);
        
        ApplyPostProcess();

        if (CurrentStressLevel <= 0.0f)
        {
			bStressActive = false;
			CurrentStressLevel = 0.0f;
        }
    }
	
}

void UCameraStressEffectComponent::TriggerStressEffect(float TargetLevel)
{
    StressLevel = FMath::Clamp(TargetLevel, 0.0f, 1.0f);

	// Démarrer le chronomètre pour la courbe
	StressStartTime = GetWorld()->GetTimeSeconds();

	bStressActive = true;
}

void UCameraStressEffectComponent::ResetStressEffect()
{
    StressLevel = 0.0f;
}

void UCameraStressEffectComponent::ApplyPostProcess()
{
    if (!Camera) return;
    auto& PP = Camera->PostProcessSettings;

    // Tunnel vision : effet de vignette
    PP.bOverride_VignetteIntensity = true;
    PP.VignetteIntensity = FMath::Lerp(0.2f, 1.0f, CurrentStressLevel);

    // Désaturation
    PP.bOverride_ColorSaturation = true;
    PP.ColorSaturation = FVector4(
        FMath::Lerp(1.0f, 0.4f, CurrentStressLevel),
        FMath::Lerp(1.0f, 0.4f, CurrentStressLevel),
        FMath::Lerp(1.0f, 0.4f, CurrentStressLevel),
        1.0f
    );

    // Atténuation de la luminosité
    PP.bOverride_AutoExposureBias = true;
    PP.AutoExposureBias = FMath::Lerp(1.0f, 0.6f, CurrentStressLevel);

    PP.bOverride_DepthOfFieldFstop = false;

    // Petit tremblement
    FRotator NewRotation = FRotator(
      FMath::FRandRange(-0.5f, 0.5f) * CurrentStressLevel * 1.5f,
      FMath::FRandRange(-0.5f, 0.5f) * CurrentStressLevel * 1.5f,
      0.0f
    );

	Camera->AddLocalRotation(NewRotation);

    // Ajustement du FOV
    Camera->SetFieldOfView(FMath::Lerp(90.0f, 100.0f, CurrentStressLevel)); // Example FOV range
}