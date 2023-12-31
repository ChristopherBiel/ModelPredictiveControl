import multiprocessing as mp
import numpy as np
import time

from astrobee import Astrobee
from mpc import MPC
from simulation import EmbeddedSimEnvironment
import user_settings

NUM_PROCESSES = 1
NUM_ITERATIONS = 2
VERBOSITY_LVL = 2       #0: best perf in end, 1: every 100 steps + best perf,  2: everything

def worker(tasksQ, resultsQ):
    verbosity = False
    if VERBOSITY_LVL == 2:
        verbosity = True
    for params in iter(tasksQ.get, 'STOP'):
        if np.mod(params['i'], 100) == 0 and VERBOSITY_LVL == 1:
            print('Running parameter set nr. ', params['i'])
        elif verbosity:
            print('Running parameter set nr. ', params['i'])
            print(params)
        score = fullRunSimu(params, verbosity=verbosity)
        resultsQ.put((score, params))

def fullRunSimu(params, verbosity=False):
    # Announce process
    # print('%s running simulation nr. %i with %i' % (mp.current_process().name, params['i'], params['Horizon']))

    # Initialise all necessary components
    abee = Astrobee(trajectory_file=params['trajectory_quat'])
    u_lim, x_lim = abee.get_limits()
    x0 = abee.get_initial_pose()

    tracking_ctl = MPC(model=abee,
                       dynamics=abee.model,
                       N=params['Horizon'],
                       trajectory_tracking=True,
                       ulb=-u_lim, uub=u_lim,
                       xlb=-x_lim, xub=x_lim,
                       params=params)
    sim_env_tracking = EmbeddedSimEnvironment(model=abee,
                                              dynamics=abee.model,
                                              controller=tracking_ctl.mpc_controller,
                                              time=80)
    tracking_ctl.set_forward_propagation()

    # Run configurations
    sim_env_tracking.run(x0)
    score = sim_env_tracking.calcScore(verbosity=verbosity)
    return score

if __name__ == '__main__':
    # Parameters are saved in a dictionary
    params = {}

    # Define paths:
    params['trajectory_quat'] = user_settings.trajectory_quat

    # Define MP setup:
    tasksQ = mp.Queue()
    resultsQ = mp.Queue()

    processes = []
    # Start
    for i in range(NUM_PROCESSES):
        process = mp.Process(target=worker, args=(tasksQ, resultsQ))
        process.start()
        processes.append(process)

    maxScore = 10
    maxParams = {}
    for i in range(NUM_ITERATIONS):
        # Check for new entries in the resultsQ
        while not resultsQ.empty():
            result = resultsQ.get()
            if result[0] > maxScore and VERBOSITY_LVL > 0:
                maxScore = result[0]
                maxParams = result[1]
                print("New max. score: %.3f with parameters" % (maxScore))
                print(maxParams)
        # Assign new tasks only when there is just one left
        while not tasksQ.empty():
            time.sleep(1)
        # Randomly choose parameters
        params['Horizon'] = 8
        Q1 = np.random.randint(1,200,3)
        Q2 = np.random.randint(1,200,3)
        Q3 = np.ones(3) * np.random.randint(1,50)
        Q4 = np.ones(3) * np.random.randint(1,50)
        params['Q'] = np.concatenate((Q1, Q2, Q3, Q4))
        params['R'] = [1,1,1,1,1,1]
        params['P'] = np.random.randint(10,100)
        params['i'] = i
        # Publish task
        tasksQ.put(params)
        # Sleep to make sure the task is published properly
        time.sleep(1)

    # After 'all' tasks are  published, we wait and then publish 'STOP'
    time.sleep(5)
    for i in range(NUM_PROCESSES):
        tasksQ.put('STOP')
        time.sleep(1)

    # Wait to close all the processes:
    for process in processes:
        process.join()

    # Fetch the final results
    while not resultsQ.empty():
        result = resultsQ.get()
        if result[0] > maxScore:
            maxScore = result[0]
            maxParams = result[1]
            print("New max. score: %.3f with parameters:" % (maxScore))
            print(maxParams)

    fullRunSimu(maxParams, verbosity=True)