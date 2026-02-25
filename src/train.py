import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

def train_CNN(model,
              train_loader,
              val_loader,
              criterion,
              optimizer,
              epochs=5,
              show_every=10):
    """
    Train a CNN model, save model states and history.

    Args:
        model       : CNN model
        train_loader: DataLoader object for the training dataset
        val_loader  : DataLoader object for the validation dataset
        criterion   : Loss function
        optimizer   : Optimizer
        epochs      : Number of epochs i.e. passes through the whole dataset
        show_every  : Frequency at which to print training progress and stats

    Returns:
        history     : Training history as a dict
    """

    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'best_acc': [0, 0.] # (epoch, best accuracy on the validation set)
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(train_loader):
            
            inputs, labels = data

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track accuracy
            __, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            
            # Print progress
            if i % show_every == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * float(correct) / total
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                
                __, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Save best model
        if val_accuracy > history['best_acc'][1]:
            history['best_acc'] = [epoch, val_accuracy]
            checkpoint = {'architecture': model.__class__.__name__,
                        'state_dict': model.state_dict(),
                        # 'optimizer' : optimizer.state_dict() # Not used here but can be activated to resume training from a given checkpoint
                        }
            torch.save(checkpoint, os.path.join(".","outputs","models", "checkpoint.pth"))

        # Update history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch {epoch+1} complete. Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    # Give checkpoint a unique file name
    model_id = datetime.datetime.now()
    model_id = model_id.strftime('%Y%m%d_%H%M%S')
    os.rename(os.path.join(".","outputs","models","checkpoint.pth"), os.path.join(".","outputs","models","checkpoint_"+checkpoint['architecture']+"_"+model_id+".pth"))

    # Save history
    np.save(os.path.join(".","outputs","models","history_"+checkpoint['architecture']+"_"+model_id+".npy"), history)
 
    return history

def plot_history(history):
    """
    Plot loss and accuracy on training and validation datasets at each epoch

    Args:
        history          : History dict (keys: 'loss', 'val_loss', 'accuracy', 'val_accuracy')

    Returns:
        None
    """
    epochs = len(history['loss'])
    fig, axs = plt.subplots(1, 2, layout='constrained')

    # Loss plot
    ax = axs[0]
    ax.plot(history['loss'], label='Training')
    ax.plot(history['val_loss'], label='Validation')
    ax.set_xlabel("Epoch")
    ax.set_xticks(np.arange(0, epochs, 2),labels=np.arange(1, epochs+1, 2))
    ax.set_title("Loss")

    # Accuracy plot
    ax = axs[1]
    ax.plot(history['accuracy'])
    ax.plot(history['val_accuracy'])
    ax.scatter(history['best_acc'][0], history['best_acc'][1], c="red")
    ax.set_xlabel("Epoch")
    ax.set_xticks(np.arange(0,epochs,2),labels=np.arange(1,epochs+1,2))
    ax.set_ylabel("%")
    ax.set_title("Accuracy")

    fig.suptitle("Training history")
    fig.legend(loc='lower left',bbox_to_anchor=(0.1, 0.1))

    # Save plot
    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join('.', 'outputs','figures', 'training_history_'+timestamp+'.png'), dpi=200, format='png')

    plt.show()
    return
