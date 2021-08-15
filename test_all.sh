# test 15Volume_planeDistance_terminateOscillate both easy and full objectives
python test_new.py --name 15Volume_planeDistance_terminateOscillate --load latest --volume_ids samp0,samp1,samp2,samp3,samp4,samp15,samp16,samp17,samp18,samp19 --n_runs 100 --fname quantitative_metrics
python test_new.py --name 15Volume_planeDistance_terminateOscillate --load latest --volume_ids samp0,samp1,samp2,samp3,samp4,samp15,samp16,samp17,samp18,samp19 --n_runs 100 --fname quantitative_metrics --easy_objective

# test 15Volume_anatomy_terminateOscillate both easy and full objectives
python test_new.py --name 15Volume_anatomy_terminateOscillate --load latest --volume_ids samp0,samp1,samp2,samp3,samp4,samp15,samp16,samp17,samp18,samp19 --n_runs 100 --fname quantitative_metrics
python test_new.py --name 15Volume_anatomy_terminateOscillate --load latest --volume_ids samp0,samp1,samp2,samp3,samp4,samp15,samp16,samp17,samp18,samp19 --n_runs 100 --fname quantitative_metrics --easy_objective