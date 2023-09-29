import multiprocessing as mp
import numpy as np
import time

from astrobee import Astrobee
from mpc import MPC
from simulation import EmbeddedSimEnvironment
import user_settings

def worker(tasksQ, resultsQ):
    for params in iter(tasksQ.get, 'STOP'):
        score = fullRunSimu(params)
        resultsQ.put(score)

def fullRunSimu(params):
    # Announce process
    print('%s running simulation nr. %i with %i' % (mp.current_process().name, params['i'], params['Horizon']))

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
    score = sim_env_tracking.calcScore(verbosity=False)
    return score

if __name__ == '__main__':
    # Parameters are saved in a dictionary
    params = {}
    NUM_PROCESSES = 6
    NUM_ITERATIONS = 100

    # Define paths:
    params['trajectory_quat'] = user_settings.trajectory_quat

    params['Horizon'] = 10
    params['Q'] = np.diag(np.array([300, 300, 300, 10, 10, 10, 100, 100, 100, 10, 10, 10]))
    params['R'] = np.diag(np.array([5, 5, 5, 50, 50, 50]))
    params['P'] = params['Q'] * 100
    
    params['i'] = 1

    # Define MP setup:
    tasksQ = mp.Queue()
    resultsQ = mp.Queue()

    processes = []
    # Start
    for i in range(NUM_PROCESSES):
        process = mp.Process(target=worker, args=(tasksQ, resultsQ)).start()
        processes.append(process)

    maxScore = 0
    for i in range(NUM_ITERATIONS):
        # Check for new entries in the resultsQ
        while not resultsQ.empty():
            newScore = resultsQ.get()
            if newScore > maxScore:
                maxScore = newScore
                print("New max. score: %.3f" % (maxScore))
        # Assign new tasks only when there is just one left
        while tasksQ.qsize() > 2:
            time.sleep(1)
        # Randomly choose parameters
        params['Horizon'] = np.random.randint(6,20)
        params['Q'] = np.diag(np.random.randint(1,300,12))
        params['R'] = np.diag(np.random.randint(1,100,6))
        params['P'] = params['Q'] * np.random.randint(10,100)
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
        processes.join()

    # Fetch the final results
    while not resultsQ.empty():
            newScore = resultsQ.get()
            if newScore > maxScore:
                maxScore = newScore
                print("New max. score: %.3f" % (maxScore))