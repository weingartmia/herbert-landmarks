import sending_data.udp_client as udp_client
import reading_camera.get_array as cam
from src.load_model import evaluate


def main():
    while True:
        frame= cam.get_frame()
        dis=evaluate(frame)
        udp_client.send()
    


if __name__ == "__main__":
    udp_client.init()
    udp_client.get_ip_address()
    cam.init()
    main()

