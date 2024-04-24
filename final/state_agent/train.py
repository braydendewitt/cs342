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

def extract_features(pstate, soccer_state, team_id):
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


def load_data(directory):
    # Initialize lists to hold all data
    all_features_list = []
    all_labels_list = []

    # Go through all pickle files
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)
            data = []
            with open(file_path, 'rb') as f:
                while True:
                    try:
                        data.append(pickle.load(f))
                    except EOFError:
                        break
            print(filename)

            # Determine which team is agent to imitate
            # Using jurgen_agent to imitate, so if pickle file starts with j, it means jurgen_agent is team1
            team_id = 0 if filename.startswith('j') else 1
            print(team_id)

            # For each state in data...
            for state in data:
                #print("State", state)
                #print("\n")
                # Get game states and actions
                team1_state = state['team1_state']
                #print("Team 1 state: ", team1_state)
                #print("\n")
                soccer_state = state['soccer_state']
                #print("Soccer state: ", soccer_state)
                #print("\n")
                team2_state = state['team2_state']
                #print("Team 2 state: ", team2_state)
                #print("\n")
                actions = state['actions']
                #print("Actions: ", actions)
                #print("\n")


                # Assign correct states based on team_id (allowing for changing the goal/side agent scores on)
                my_imitation_agent_states = team1_state if team_id == 0 else team2_state
                #print("My agent states: ", my_imitation_agent_states)
                #print("\n")
                opponent_agent_states = team2_state if team_id == 0 else team1_state

                # For each of our agent's karts in each state of the data...
                for i, player_state in enumerate(my_imitation_agent_states):
                    if i < len(actions) and i < len(opponent_agent_states):
                        #print("i value: ", i)
                        #print("\n")
                        #print("current kart/player state: ", player_state)
                        #print("\n")
                        # Get kart's features
                        features = extract_features(player_state, soccer_state, team_id)
                        #print("New features extracted: ", features)
                        #print("\n")
                        all_features_list.append(features)
                        #print("Full feature list: ", all_features_list)
                        #print("\n")
                        # Get kart's actions
                        if team_id == 0:
                            if i == 0:
                                kart_id = 0
                            if i == 1:
                                kart_id = 2
                        if team_id == 1:
                            if i == 0:
                                kart_id = 1
                            if i == 1:
                                kart_id = 3
                        action = actions[kart_id]
                        #print("All actions: ", actions)
                        #print("\n")
                        #print("Kart ID: ", kart_id)
                        #print("\n")
                        #print("Selected actions from indexing: ", action)
                        #print("\n")
                        # Get kart's corresponding labels
                        labels = torch.tensor([action.get('acceleration'), action.get('steer'), action.get('brake')])
                        #print("Current labels - should match selected actions: ", labels)
                        #print("\n")
                        all_labels_list.append(labels)
                        #print("Full labels list: ", all_labels_list)
                        #print("\n")
    

    # Stack all features and labels into tensor
    all_features_tensor = torch.stack(all_features_list)
    all_labels_tensor = torch.stack(all_labels_list)

    print("Features tensor shape:", all_features_tensor.shape)
    print("Labels tensor shape:", all_labels_tensor.shape)

    return all_features_tensor, all_labels_tensor

def train(args):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Initialize model
    model = ImitationModel().to(device)

    # Loss function and optimizer
    #loss_fn = nn.L1Loss().to(device)
    loss_fn = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
                                          
    ## Load in data
    features, actions = load_data('../validated_pickle_files')
    dataset = TensorDataset(torch.tensor(features, dtype = torch.float32), torch.tensor(actions, dtype = torch.float32))
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

    # Initialize loss
    current_loss = float('inf')

    # Training loop...
    for epoch in range(args.epochs):
        model.train()
        total_loss_for_epoch = 0
        for inputs_features, labels_actions in dataloader:

            # Send to device
            inputs_features = inputs_features.to(device)
            labels_actions = labels_actions.to(device)

            # Get prediction
            output = model(inputs_features)

            # Debugging
            """   if epoch % 20 == 0:
                print("Input features: ", inputs_features)
                print("\n")            
                print("Predicted output: ", output)
                print("\n")
                print("Jurgen labels: ", labels_actions)
                print("\n")
            """

            # Calculate loss
            loss_val = loss_fn(output, labels_actions)
            total_loss_for_epoch += loss_val.item()

            # Zero gradient
            optimizer.zero_grad()

            # Backward pass and optimizer step
            loss_val.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        print(f"Epoch {epoch+1}/{args.epochs}, Total Loss for Epoch: {total_loss_for_epoch}")

        # Save model
        if float(total_loss_for_epoch) < current_loss:
            current_loss = float(total_loss_for_epoch)
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, 'state_agent.pt')
            print(f"Saving model at epoch {epoch + 1} with loss of {total_loss_for_epoch}")

        # Save model every 10 epochs if loss has improved
        if epoch == 0:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch0.pt')
        if epoch == 10:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch10.pt')        
        if epoch == 20:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch20.pt')
        if epoch == 30:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch30.pt')
        if epoch == 40:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch40.pt')        
        if epoch == 50:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch50.pt')            
        if epoch == 60:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch60.pt')
        if epoch == 70:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch70.pt')        
        if epoch == 80:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch80.pt')            
        if epoch == 90:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch90.pt')
        if epoch == 100:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch100.pt')        
        if epoch == 110:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch110.pt')
        if epoch == 120:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch120.pt')
        if epoch == 130:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch130.pt')        
        if epoch == 140:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch140.pt')
        if epoch == 150:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch150.pt')
        if epoch == 160:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch160.pt')        
        if epoch == 170:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch170.pt')            
        if epoch == 180:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch180.pt')
        if epoch == 190:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch190.pt')        
        if epoch == 200:
            scripted_model_epoch = torch.jit.script(model)
            torch.jit.save(scripted_model_epoch, 'state_agent_epoch200.pt')            

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