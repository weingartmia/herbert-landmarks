import socket

local_port= 8888
buffer_size= 1024

sock= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

def init():
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1) 
    sock.bind(('', local_port))
    print("UDP server : {}:{}".format(get_ip_address(),local_port))

def main():
    while True:
        data, addr = sock.recvfrom(1024) 
        sock.sendto("RPi received OK",addr)

def send(distance,rotation):
    data, addr = sock.recvfrom(1024) 
    sock.sendto((distance,rotation),addr)
    


def get_ip_address():
    ip_address = ''
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8",80))
    ip_address = s.getsockname()[0]
    s.close()
    return ip_address

if __name__=="__main__":
    init()
    main()