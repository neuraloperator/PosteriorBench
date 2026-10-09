from posteriorbench.methods.base import MethodAdapter


METHODS = {
    "fundps",
    "diffusionpde",
    "funddps",
    "ddis",
    "fundiff",
    "eci",
    "esmda",
    "mcdropout",
}


def create_method(name: str, **kwargs) -> MethodAdapter:
    key = name.lower()
    if key == "fundps":
        from posteriorbench.methods.fundps import FunDPSAdapter

        return FunDPSAdapter(**kwargs)
    if key == "diffusionpde":
        from posteriorbench.methods.diffusionpde import DiffusionPDEAdapter

        return DiffusionPDEAdapter(**kwargs)
    if key == "funddps":
        from posteriorbench.methods.funddps import FunDDPSAdapter

        return FunDDPSAdapter(**kwargs)
    if key == "ddis":
        from posteriorbench.methods.ddis import DDISAdapter

        return DDISAdapter(**kwargs)
    if key == "fundiff":
        from posteriorbench.methods.fundiff import FunDiffAdapter

        return FunDiffAdapter(**kwargs)
    if key == "eci":
        from posteriorbench.methods.eci import ECIAdapter

        return ECIAdapter(**kwargs)
    if key == "esmda":
        from posteriorbench.methods.esmda import ESMDAAdapter

        return ESMDAAdapter(**kwargs)
    if key == "mcdropout":
        from posteriorbench.methods.mcdropout import MCDropoutAdapter

        return MCDropoutAdapter(**kwargs)
    raise ValueError(f"Unknown method '{name}'. Available: {sorted(METHODS)}")


__all__ = ["create_method"]
