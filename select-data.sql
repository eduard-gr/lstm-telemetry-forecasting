

select obd from fm_volvo_odb_telemetry where (obd::jsonb ? 'EngineLoad') = true limit 2

select json_agg(jsonb_build_object('Date',fmc_date::timestamp) || obd::jsonb  order by fmc_date ) from fm_volvo_odb_telemetry where (obd::jsonb ? 'EngineLoad') = true limit 2

insert into fm_volvo_odb_telemetry
select
	fme.fmc_id,
	min(fmc_date) as fmc_date,
	json_object_agg(
		case
			when fme_type = 239 then 'Ignition'
			when fme_type = 30 then 'DTC'

			when fme_type = 37 then 'VehicleSpeed'
			when fme_type = 31 then 'EngineLoad'
			when fme_type = 36 then 'EngineRPM'
			when fme_type = 41 then 'ThrottlePosition'

			when fme_type = 32 then 'CoolantTemperature'
			when fme_type = 39 then 'IntakeAirTemperature'
			when fme_type = 53 then 'AmbientAirTemperature'

			when fme_type = 35 then 'IntakeMAP'  --(Intake manifold absolute pressure )
			when fme_type = 40 then 'MAF' -- MAF Air flow rate
--			when fme_type = 42 Runtime since engine start
--			when fme_type = 43 Distance Traveled MIL On
			when fme_type = 45 then 'DirectFuelRailPressure'
			when fme_type = 46 then 'CommandedEGR'
			when fme_type = 47 then 'EGRError'
			when fme_type = 48 then 'FuelLevel'
--			when fme_type = -- 49 Distance Since Codes Clear
			when fme_type = 50 then 'BarometicPressure'
			when fme_type = 51 then 'ControlModuleVoltage'
--			when fme_type = 55 then 'Time Since Codes Cleared'
		end,
		fme_value) as obd
from
	--fm_event
	--
	(select fmc_id from "partition".fm_event_2020_11 where fmobj_id = 1274 and fme_type = 239 and fme_value = 1) fmc
	join "partition".fm_event_2020_11 fme on
		fme.fmc_id = fmc.fmc_id
		and fme.fme_type in (30,37,31,36,41,32,39,53,35,40,45,46,47,48,50,51,239)

group by
	fme.fmc_id