import json
import astropy.units as u

def load_config(filepath="config.json"):
    """Load configuration from a JSON file, re-attach Astropy units, 
    and return a dictionary for easy access across your pipeline.
    """
    with open(filepath, "r") as f:
        config = json.load(f)
    
    # Re-attach Astropy units to parameters that require them
    unit_mappings = {
        "snap_radius_arcsec": u.arcsec,
        "isolation_radius_arcsec": u.arcsec,
        "wavelength_tolerance_um": u.um,
        "cutout_size_arcmin": u.arcmin,
        "spectrum_cutout_size_arcsec": u.arcsec,
        "pixel_arcsec": u.arcsec,
    }
    
    for key, unit in unit_mappings.items():
        if key in config:
            config[key] = config[key] * unit
            
    # Convert tuple parameters back from JSON lists if necessary
    if "annulus_pix" in config and isinstance(config["annulus_pix"], list):
        config["annulus_pix"] = tuple(config["annulus_pix"])
        
    return config
