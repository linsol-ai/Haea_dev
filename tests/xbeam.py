import xarray
import pandas as pd
import numpy as np
import time
import xarray_beam

drop_vars = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature', 'angle_of_sub_gridscale_orography', 'anisotropy_of_sub_gridscale_orography', 'boundary_layer_height', 'geopotential_at_surface', 'high_vegetation_cover', 'lake_cover', 'land_sea_mask', 'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation', 'low_vegetation_cover', 'mean_sea_level_pressure', 'mean_surface_latent_heat_flux', 'mean_surface_net_long_wave_radiation_flux', 'mean_surface_net_short_wave_radiation_flux', 'mean_surface_sensible_heat_flux', 'mean_top_downward_short_wave_radiation_flux', 'mean_top_net_long_wave_radiation_flux', 'mean_top_net_short_wave_radiation_flux', 'mean_vertically_integrated_moisture_divergence', 'potential_vorticity', 'sea_ice_cover', 'sea_surface_temperature', 'slope_of_sub_gridscale_orography', 'snow_depth', 'soil_type',  'standard_deviation_of_filtered_subgrid_orography', 'standard_deviation_of_orography', 'surface_pressure', 'total_cloud_cover', 'total_column_water', 'total_column_water_vapour', 'total_precipitation_12hr', 'total_precipitation_24hr', 'total_precipitation_6hr', 'type_of_high_vegetation', 'type_of_low_vegetation', 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4']
variable = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']

# 파이프라인 실행
if __name__ == '__main__':
    start = time.time()
    ds = xarray.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr', 
                          consolidated=True, 
                          chunks=None,
                          drop_variables=drop_vars
                          )
    new_dataset = ds.sel(time=slice('2021-01-01', '2022-01-01'), latitude=slice(39.0, 32.2), longitude=slice(124.2, 131))
    
    print(new_dataset[].values)
    end = time.time()
    print(f"{end - start:.5f} sec")
