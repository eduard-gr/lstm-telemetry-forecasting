drop table fm_volvo_odb_telemetry

create table fm_volvo_odb_telemetry as
SELECT
	fmc.fmc_date,
	telemetry.*
FROM
	crosstab(
		'select
			fmc_id,
			fme_type,
			fme_value
		from
			fm_event
		where
			fmobj_id = 1274
			and fme_type in (239, 200, 240, 30, 24, 31, 36, 41, 32, 39, 53, 35, 40, 45, 46, 47, 48, 50, 51)
		group by
			fmc_id,
			fme_type,
			fme_value
		order by
			min(fmc_date),
			fmc_id',
		'select
			unnest(ARRAY[239, 200, 240, 30, 24, 31, 36, 41, 32, 39, 53, 35, 40, 45, 46, 47, 48, 50, 51])') as telemetry(
			fmc_id int8,
			ignition int4,
			sleep_mode int4,
			movement int4,

			dtc int4,
			speed int4,
			engine_load int4,
			engine_rpm int4,
			throttle_position int4,

			coolant_temperature int4,
			intake_air_temperature int4,
			ambient_air_temperature int4,

			intake_map int4,
			maf int4,
			direct_fuel_rail_pressure int4,
			commanded_egr int4,
			egr_error int4,
			fuel_level int4,
			barometic_pressure int4,
			control_module_voltage int4)
	join fleetmanagement.fm_coordinate fmc on fmc.fmc_id = telemetry.fmc_id