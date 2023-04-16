import socket
import pickle
from Ur_robot_controls import Ur3Control

def GetConfirmation():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    client_socket.sendall(b'request')
    serialized_data = client_socket.recv(4096)
    client_socket.close()
    ConfirmationMessage = pickle.loads(serialized_data)
    return ConfirmationMessage

# Create a Ur5Control object
ur3 = Ur3Control()

# Print the current joint positions
Current_positions = ur3.joint_states_callback(rospy.wait_for_message("/joint_states", JointState))
ConfirmationMessage = GetConfirmation()
i = int(ConfirmationMessage)
#Save the current joint positions as a .npz as JointPos_xxx.npz where npz is the number of the image where i the image number, with 3 digits
np.savez("JointPos_" + str(i).zfill(3), Current_positions)

print("Done")