/*select
    json_agg(jsonb_build_object('Date',fmc_date::timestamp) || obd::jsonb  order by fmc_date )
from
    fm_volvo_odb_telemetry
where
    fmc_date between '2020-11-22 11:22:02' and '2020-11-22 15:26:50'*/

select
    fmc_date::timestamp,
    engine_rpm,
    ignition,
    movement,
    fmmu_type,
    speed,
    engine_load,
    throttle_position,
    coolant_temperature,
    intake_air_temperature,
    ambient_air_temperature,
    intake_map,
    maf,
    direct_fuel_rail_pressure,
    commanded_egr,
    egr_error,
    fuel_level,
    barometic_pressure,
    control_module_voltage
from
    fm_volvo_odb_telemetry
/*where
    engine_rpm is not null*/
/*where
    date_trunc('month', fmc_date) = '2020-01-01 00:00:00'*/
order by
    fmc_date


