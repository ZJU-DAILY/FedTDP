import argparse
from party.server import Server
from party.client import Client
import secretflow as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--task', type=str, default='noise filtering')
    parser.add_argument('--dataset', type=str, default='geolife')
    parser.add_argument('--data', type=str, default='dataset/')
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--clients_num', type=int, default=8)
    parser.add_argument('--drop_out', type=float, default=0.2)
    parser.add_argument('--max_embed_length', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--output_size', type=int, default=32)
    parser.add_argument('--frozen', type=int, default=5)
    parser.add_argument('--m', type=float, default=0.25)
    parser.add_argument('--out', type=int, default=20)
    parser.add_argument('--slm', type=str, choices=['gpt3-small', 'gpt2-small', 't5-small'])
    parser.add_argument('--llm', type=str, choices=['llama', 'gpt3', 'qwen'])
    parser.add_argument('--address', type=str, default='localhost')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json')
    args = parser.parse_args()
    cs = ['client' + str(i) for i in range(args.clients_num)]
    sf.init(parties=['server'] + cs, address=args.address)
    server = Server(args=args, party=sf.SPU(party='server'))
    clients = [Client(args=args, client_num=i, server=server, party=sf.SPU(party=cs[i])) for i in
               range(args.clients_num)]
    server.clients = clients
    for client in clients:
        client.clients = clients
    server.secret_sharing()
    server.train()
