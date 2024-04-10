import socket

UDP_IP = '10.128.73.102'
UDP_PORT = 5005
# 3 - right
MESSAGE = '1125010'

print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
print("message: %s" % MESSAGE)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP   

sock.sendto(MESSAGE.encode('UTF-8'), (UDP_IP, UDP_PORT))
    