from typing import List, Dict, Any

class HardwareMatrixTester:
    """
    v2.0 Hardware Matrix Tester: Explicitly asserts backward/forward hardware compatibility 
    and validates deployment metadata against supported target profiles.
    """
    def __init__(self):
        # Matrix mapping CUDA capability to available optimizations
        self.supported_sm_profiles = {
            "sm_80": ["fp16", "tf32"],
            "sm_86": ["fp16", "tf32", "int8"],
            "sm_90": ["fp16", "tf32", "int8", "fp8", "wgmma", "tma"]
        }

    def validate_capability_matrix(self, target_sm: str, required_optimizations: List[str]) -> bool:
        """
        Validates whether target SM capability supports the required optimizations.
        """
        if target_sm not in self.supported_sm_profiles:
            raise ValueError(f"Unsupported target hardware profile: {target_sm}")
            
        supported_features = self.supported_sm_profiles[target_sm]
        for opt in required_optimizations:
            if opt not in supported_features:
                return False
                
        return True

    def cross_compile_profiles(self, schema: Dict[str, Any]) -> List[str]:
        """
        Determines the dynamic array of compatible cross-compilation profiles 
        to freeze into the crystallized YAML frontmatter.
        """
        hardware_meta = schema.get("hardware_requirements", {})
        quantization_meta = hardware_meta.get("quantization", {})
        
        # Determine the lowest baseline target based on needed precision
        baseline_opts = ["fp16"]
        if "int8" in str(quantization_meta).lower():
            baseline_opts.append("int8")
        if "fp8" in str(quantization_meta).lower():
            baseline_opts.append("fp8")

        compatible_targets = []
        for sm, features in self.supported_sm_profiles.items():
            if all(opt in features for opt in baseline_opts):
                compatible_targets.append(sm)

        return compatible_targets
