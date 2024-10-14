
from scapy.all import sniff, show_interfaces
import pandas as pd
import datetime
import threading
import time

def packet_handler(packet, packet_data):
    # Timestamp
    timestamp = datetime.datetime.fromtimestamp(packet.time).strftime('%Y-%m-%d %H:%M:%S')

    # Initialize feature values
    src_ip = packet[0][1].src if packet.haslayer('IP') else None
    dst_ip = packet[0][1].dst if packet.haslayer('IP') else None
    protocol = packet[0][1].proto if packet.haslayer('IP') else None

    # Convert protocol number to name
    if protocol:
        protocol = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}.get(protocol, str(protocol))

    # Source and Destination Ports
    src_port = packet[0][1].sport if packet.haslayer('TCP') or packet.haslayer('UDP') else None
    dst_port = packet[0][1].dport if packet.haslayer('TCP') or packet.haslayer('UDP') else None

    # Packet Length
    pkt_len = len(packet)

    # Append to packet data list
    packet_data.append({
        'Timestamp': timestamp,
        'Source_IP': src_ip,
        'Destination_IP': dst_ip,
        'Protocol': protocol,
        'Source_Port': src_port,
        'Destination_Port': dst_port,
        'Packet_Length': pkt_len
    })

def print_sniff_status(packet_data, count):
    while len(packet_data) <= count:
        print(f'\rCaptured {len(packet_data)} packets so far...', end='')
        time.sleep(0.5)

def capture_packets(networkInterface, count, output_csv):
    packet_data = []

    print(f'Starting capture of {count} packets on interface: {networkInterface}')
    
    # Start a thread to print the sniff status
    status_thread = threading.Thread(target=print_sniff_status, args=(packet_data, count))
    status_thread.daemon = True
    status_thread.start()
    
    # Start sniffing packets
    sniff(
        iface=networkInterface,
        count=count,
        prn=lambda packet: packet_handler(packet, packet_data),
        store=False
    )
        
    print('\nPacket capture complete.')
    print('Exporting to CSV...')

    # Create a DataFrame and export to CSV
    df = pd.DataFrame(packet_data)
    df.to_csv(output_csv, index=False)

    print(f'Data exported to {output_csv}')

print('### START ###')
# show_interfaces()

networkInterface = 'Realtek 8822CE Wireless LAN 802.11ac PCI-E NIC'
packet_count = 1000  # Number of packets to capture
output_file = 'datasets/homeMadeDataset.csv'

capture_packets(networkInterface, packet_count, output_file)

print('### END ###')

