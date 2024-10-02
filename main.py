from pymavlink import mavutil
from math import isclose
import time
from rl_landing.controller import load_system

import argparse

from random import randint

import os

PKG_PATH = os.path.join(os.getcwd(),'src','rl_landing','rl_landing')

""" Parsing arguments """
parser = argparse.ArgumentParser()

parser.add_argument('--mission', type=str, required=False, default='land', choices=['land','autonomous','detection'])
parser.add_argument('--takeoff_altitude', type=float, required=False, default=6.0)
parser.add_argument('--control_period', type=float, required=False, default=3.0)
parser.add_argument('--detection_period', type=float, required=False, default=0.1)
parser.add_argument('--system_name', type=str, required=False, choices=['dummy','dqn','lander','vital'], default='dummy')

parser.add_argument('--train', action='store_true', help = "It trains from scratch or trains from given model if --resume is True")
parser.add_argument('--test', action='store_true', help = "It tests the given model")
parser.add_argument('--freeze', action='store_true', help = "If activated, the detector is frozen (does not train)")
parser.add_argument('--resume', action='store_true', help="It resumes training of provided model in --model_path")
parser.add_argument('--model_path', type=str, required=False, default='', help="This is the path to the model to test (.ckpt or .pt)")
parser.add_argument('--run_path', type=str, required=False, default='', help="This is the path to the run to resume training")
parser.add_argument('--config_file', type=str, required=False, default='', help="This is the configuration containing the parameters in a .json file of the attributes of the model to test")

ap = parser.parse_args()

if ap.mission == "autonomous":
    if ap.train + ap.test + ap.resume > 1:
        print('You can not activate simultaneous --train, --test and --resume flags. Activate just one.')
        exit()
    if ap.train: 
        if not ap.config_file:
            print('Not a provided model_path to train. Fill the flag --model_path [PATH_TO_MODEL] with the model to train')
            exit()
    elif ap.test:
        if not ap.model_path:
            print('Not a provided model_path to test. Fill the flag --model_path [PATH_TO_MODEL] with the model to test')
            exit()
        elif not ap.config_file:
            print('Not a provided config file to test. Fill the flag --config [CONFIG in .json] with the parameters of the model to test')
            exit()
    elif ap.resume:
        if not ap.run_path:
            print('Not a provided run_path to resume training. Fill the flag --run_path [RUN_OF_TRAIN] with the run to resume')
            exit()
        elif not ap.config_file:
            # TODO to resume we should only be loading saved config
            print('Not a provided config file to resume. Fill the flag --config [CONFIG in .json] with the parameters of the model to test')
            exit()

THRUSTER_RF = 1
THRUSTER_LF = 2
THRUSTER_RB = 3
THRUSTER_LB = 4

DISABLE_FUNCTION = 0

"""
Change servo function. As args it is an (int) servo ID, a function number and a mavlink connection object
e.g.,: set_servo_function(1,0), disables autopilot from servo 1
"""
def set_servo_function(servo_id, function_num, conn):
    print(f'Setting servo {servo_id} to function {function_num}')
    conn.mav.param_set_send(conn.target_system, 
                                  conn.target_component, 
                                  ("SERVO" + str(servo_id) + "_FUNCTION").encode(), 
                                  function_num, 
                                  mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                                  )

"""
Creates and returns a mavlink command to change the servo's PWM. 
As args it is an (int) servo ID, the PWM value and a mavlink connection object
"""
def thruster_cmd(servo_id, pwm, conn):
    print(f'Sending {pwm} PWM for thruster {servo_id}')
    cmd = conn.mav.command_long_encode(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        servo_id,
        pwm,
        0,0,0,0,0
    )
    return cmd

"""
Sets copter mode. As args there is the mavlink connection and the desired mode
"""
def set_mode(connection, mode):
    if mode not in connection.mode_mapping():
        print(f'Unknown mode {mode}')
        #rospy.signal_shutdown('Wrong mode set')
        exit()
    connection.mav.set_mode_send(connection.target_system, 
                           mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                           connection.mode_mapping()[mode]
                           )
    """ Check if command is successful """
    while True:
        resp = connection.recv_match(type='HEARTBEAT', blocking=True).custom_mode # waits for response
        if mavutil.mode_mapping_acm.get(resp) != mode:
            continue
        print(f'Successfully changed to mode {mode}')
        break

"""
Arm the vehicle and waits for arming finished
"""
def arm(connection):
    print('Waiting for arm')
    connection.arducopter_arm()
    while True:
        connection.wait_heartbeat()
        connection.arducopter_arm()
        print('Sent arm request (each 3 seconds)')
        time.sleep(3)
        if connection.motors_armed():
            break
    print('Vehicle armed!')

def disarm(connection, timeout=15):
    st = time.time()
    """ Start loop of waiting for disarm """
    while True:
        m = connection.wait_heartbeat()
        if not connection.motors_armed():
            print('Disarmed!')
            return
        elif (time.time() - st) > timeout: 
            """ If time exceeds and not disarmed, break the loop and send command manually """
            break

    """ If did not disarm, send disarm command """
    connection.mav.command_long_send(
    connection.target_system,    # target_system
    connection.target_component, # target_component
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, # Command
    0,                           # Confirmation
    0,                           # Disarm (param1)
    0, 0, 0, 0, 0, 0             # Unused parameters
    )

    ack = connection.recv_match(type='COMMAND_ACK', blocking=True)
    print("After sending manual disarm command, ACK received: %s" % ack)

""" 
Blocks until received mavlink message of type GLOBAL_POSITION_INT.
As args it is the mavlink connection object.
It returns the gps location message
"""
def get_gps_position(connection):
    connection.wait_heartbeat()
    while True:
        msg = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        return msg

"""
Function that blocks until altitude in meters is reached.
It checks altitude each 500ms
Reaching tolerance is 0.3m by default
@TODO put timeout and quit
"""
def wait_altitude(connection, altitude, tolerance=0.5):
    print('Waiting for altitude reaching')
    st_timeout = time.time()
    while True:
        cur_alt = get_gps_position(connection).relative_alt / 1e3
        print(f'Remaining distance to target altitude: {round(abs(cur_alt-altitude),ndigits=3)}')
        if isclose(cur_alt, altitude, abs_tol=tolerance):
            print(f'Altitude of {cur_alt} reached!')
            return
        elif (time.time() - st_timeout) > 15:
            print('Timeout of altitude check exceeded, finished takeoff. ')
            return
        time.sleep(0.5)

"""
Orders the takeoff to the altitude given in the argument
"""
def takeoff(connection, altitude, block=False):
    print(f'Sending takeoff cmd to {altitude}m')
    connection.mav.command_long_send(
        connection.target_system,        # Target system ID
        connection.target_component,     # Target component ID
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,  # Command ID for takeoff
        0,                           # Confirmation
        0,                           # Param 1 (ignored by takeoff command)
        0,                           # Param 2 (ignored by takeoff command)
        0,                           # Param 3 (ignored by takeoff command)
        0,                           # Param 4 (ignored by takeoff command)
        0,                           # Param 5 (latitude, not used)
        0,                           # Param 6 (longitude, not used)
        altitude                     # Param 7 (takeoff altitude)
    )
    if block:
        wait_altitude(connection, altitude)

def takeoff_mission(connection, altitude):
    """ Set guided """
    set_mode(connection, 'GUIDED')

    """ Arm """
    arm(connection)

    """ Takeoff """
    takeoff(connection, altitude, block=True)

def land_mission(connection):
    """ Set guided """
    set_mode(connection, 'LAND')
    disarm(connection, timeout=15)

def set_thrusters(connection, rf_pwm, lf_pwm, rb_pwm, lb_pwm):
    connection.mav.send(thruster_cmd(THRUSTER_RF, rf_pwm, connection))
    connection.mav.send(thruster_cmd(THRUSTER_LF, lf_pwm, connection))
    connection.mav.send(thruster_cmd(THRUSTER_RB, rb_pwm, connection))
    connection.mav.send(thruster_cmd(THRUSTER_LB, lb_pwm, connection))

def set_autopilot(connection, turn_on=True):
    if turn_on:
        set_servo_function(THRUSTER_RF, 33, connection)
        set_servo_function(THRUSTER_LF, 34, connection)
        set_servo_function(THRUSTER_RB, 35, connection)
        set_servo_function(THRUSTER_LB, 36, connection)
        print('Turned on autopilot')
    else:
        set_servo_function(THRUSTER_RF, DISABLE_FUNCTION, connection)
        set_servo_function(THRUSTER_LF, DISABLE_FUNCTION, connection)
        set_servo_function(THRUSTER_RB, DISABLE_FUNCTION, connection)
        set_servo_function(THRUSTER_LB, DISABLE_FUNCTION, connection)
        print('Turned off autopilot')

"""
Mission that just launches a trained detector and publishes the bounding box location
"""
def detection_mission(connection):
    """ Land first """
    land_mission(connection)

    """ Set guided """
    set_mode(connection, 'GUIDED')

    """ Arm """
    arm(connection)

    """ Takeoff """
    takeoff(connection, altitude=3, block=True)

    """ Initialize controller """
    detector = load_system(
            name=ap.system_name,
            mission=ap.mission,
            to_train=ap.train, 
            to_test=ap.test, 
            to_resume=ap.resume, 
            model_path=ap.model_path, 
            freeze=ap.freeze,
            config_file=ap.config_file,
            pkg_path=PKG_PATH
    )
    print(f'Initialized detector {ap.system_name}')

    #detector.set_gz_model_pose("artuga_0", detector.landing_target_position) # Set artuga to new position

    """ Start control cycle """
    try:
        while True:
            """ Call detector, meaning asking the detector to infer marker location """
            report = detector() # report return has marker detected location

            """ Set pace """
            time.sleep(ap.detection_period)
            
    except KeyboardInterrupt:
            #controller.finish() TODO remove
            land_mission(connection)
            print('Interrupted mission')

def autonomous_mission(connection):
    # """ Land first """
    # land_mission(connection)

    # """ Set guided """
    # set_mode(connection, 'GUIDED')

    # """ Arm """
    # arm(connection)

    # """ Takeoff """
    # takeoff(connection, altitude=3, block=True)

    """ Initialize controller """
    controller = load_system(
            name=ap.system_name, 
            to_train=ap.train, 
            to_test=ap.test, 
            to_resume=ap.resume, 
            model_path=ap.model_path,
            run_path=ap.run_path,
            mission=ap.mission,
            config_file=ap.config_file,
            freeze=ap.freeze,
            pkg_path=PKG_PATH
    )
    print(f'Initialized controller {ap.system_name}')

    """ Start control cycle """
    try:
        while True:
            """ Call controller, meaning, rolling an RL interaction """
            report = controller() # report return has e.g., (episode ended: bool, ending cause: landed, new_target_pose: np.array)
            exit() # TODO here
            if report[0].item(): # if terminated episode from controller, restart UAV to new position
                print('Ended episode. Resetting UAV and marker')

                controller.set_gz_model_pose('artuga_0', report[2])

                land_mission(connection)

                takeoff_mission(connection, altitude=randint(1,7)) # put random height and position

            """ Set pace """
            time.sleep(ap.control_period) # TODO check if state changed instead of periodic control
            
    except KeyboardInterrupt:
            controller.finish()
            land_mission(connection)
            print('Interrupted mission')

def main():
    """ Start connection with vehicle """
    connection = mavutil.mavlink_connection('127.0.0.1:14550')
    connection.wait_heartbeat()
    print("Heartbeat from system (system %u component %u)" % (connection.target_system, connection.target_component))

    """ Takeoff mission """
    if ap.mission == "takeoff": takeoff_mission(connection, ap.takeoff_altitude)
    
    """ Land mission """
    if ap.mission == "land": land_mission(connection)

    """ Control mission """
    if ap.mission == "autonomous": autonomous_mission(connection)

    """ Detection mission """
    if ap.mission == "detection": detection_mission(connection)

if __name__ == '__main__':
    main()