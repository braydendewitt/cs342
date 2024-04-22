import torch
import numpy as np
from model import ImitationModel
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from os import path
import pickle
import os

## Taken from jurgen_agent code...

def limit_period(angle):
    # turn angle into -1 to 1 
    return angle - torch.floor(angle / 2 + 0.5) * 2 

def extract_features(pstate, soccer_state, opponent_state, team_id):
    # features of ego-vehicle
    kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # features of soccer 
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0]) 

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

    # features of score-line 
    goal_line_center = torch.tensor(soccer_state['goal_line'][(team_id+1)%2], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)

    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, 
        goal_line_center[0], goal_line_center[1], kart_to_puck_angle_difference, 
        puck_center[0], puck_center[1], puck_to_goal_line[0], puck_to_goal_line[1]], dtype=torch.float32)

    return features 

def load_data(file_path):
    # Open pkl file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Initialize list
    features_list = []
    labels_list = []
    
    # Get team1 states
    team1_states = data['team1_state']
    # Get soccer state
    soccer_state = data['soccer_state']
    # Get actions
    actions = data['actions']

    # For each...
    for i, pstate in enumerate(team1_states):
        if i < len(actions): 
            team_id = 0
            opponent_states = data['team2_state'] # Get opponent state
            
            # Extract features
            features = extract_features(pstate, soccer_state, opponent_states[i], team_id)
            features_list.append(features.detach())  # Add to feature list 
            
            # Get labels/actions
            action = actions[i]
            labels = torch.tensor([action.get('acceleration', 0), action.get('steer', 0), action.get('brake', 0)])
            labels_list.append(labels)

    # Stack all features and labels to create tensors
    features_tensor = torch.stack(features_list)
    labels_tensor = torch.stack(labels_list)

    return features_tensor, labels_tensor

def new_load_data(directory):
    # Initialize lists to hold all data
    all_features_list = []
    all_labels_list = []

    # Go through all pickle files
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

                # Temp list for current file
                features_temp_list = []
                labels_temp_list = []

                # Get game states and actions
                team1_states = data['team1_state']
                soccer_state = data['soccer_state']
                actions = data['actions']
                opponent_states = data['team2_state']

                # For each...
                for i, pstate in enumerate(team1_states):
                    if i < len(actions) and i < len(opponent_states):
                        team_id = 0

                        # Get features
                        features = extract_features(pstate, soccer_state, opponent_states[i], team_id)
                        features_temp_list.append(features)

                        # Get labels/actions
                        action = actions[i]
                        labels = torch.tensor([action.get('acceleration', 0), action.get('steer', 0), action.get('brake', 0)])
                        labels_temp_list.append(labels)
                
                # Add on to overall list
                all_features_list.extend(features_temp_list)
                all_labels_list.extend(labels_temp_list)

    # Stack all features and labels into tensor
    all_features_tensor = torch.stack(all_features_list)
    all_labels_tensor = torch.stack(all_labels_list)

    return all_features_tensor, all_labels_tensor

def train(args):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Initialize model
    model = ImitationModel().to(device)

    # Loss function and optimizer
    loss_fn = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)


    ## Load in data
    features, actions = new_load_data('../pkl_files')
    dataset = TensorDataset(torch.tensor(features, dtype = torch.float32), torch.tensor(actions, dtype = torch.float32))
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

    # Initialize loss
    current_loss = float('inf')

    # Training loop...
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for inputs_features, labels_actions in dataloader:
            
            # Send to device
            inputs_features = inputs_features.to(device)
            labels_actions = labels_actions.to(device)

            # Get prediction
            output = model(inputs_features)

            # Calculate loss
            loss_val = loss_fn(output, labels_actions)
            total_loss += loss_val.item()

            # Zero gradient
            optimizer.zero_grad()

            # Backward pass and optimizer step
            loss_val.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss}")

        # Save model
        if float(total_loss) < current_loss:
            current_loss = float(total_loss)
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, 'state_agent.pt')
            print(f"Saving model at epoch {epoch + 1} with loss of {total_loss}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Put custom arguments here
    parser.add_argument('--epochs', type=int, default=25) # Number of epochs
    parser.add_argument('--batch_size', type=int, default=32) # Batch size
    parser.add_argument('--lr', type=float, default=0.001) # Learning rate
    parser.add_argument('--wd', type=float, default=1e-4) # Weight decay
    args = parser.parse_args()
    train(args)
