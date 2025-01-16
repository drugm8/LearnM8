# LearnM8
Active learning package for docking/consensus score prediction

to start runner : nohup python RUNNER.py >/dev/null 2>&1 &
or: nohup python RUNNER.py >runner.log 2>&1 &


tensorboard --logdir tb_logs to get live information on training

rsync -avzP maba00001@conduit.cs.uni-saarland.de:/home/maba00001/LearnM8/cpu_results ./