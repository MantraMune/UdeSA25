#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Camera/CameraComponent.h"

#include "CameraStressEffectComponent.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class TESTCAM_API UCameraStressEffectComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UCameraStressEffectComponent();

    UFUNCTION(BlueprintCallable)
    void TriggerStressEffect(float TargetLevel);

    UFUNCTION(BlueprintCallable)
    void ResetStressEffect();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Stress")
    float StressLevel; // 0 = calme, 1 = maximum stress

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Stress")
	UCurveFloat* StressCurve;
    
protected:
    virtual void BeginPlay() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:
    UCameraComponent* Camera;
    float CurrentStressLevel;
    void ApplyPostProcess();
    float StressStartTime = -1.0f;
	bool bStressActive = false;
};
