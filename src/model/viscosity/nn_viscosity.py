import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel
import joblib

import src.data_preparation.descriptor
import src.data_preparation.preparation as preparation


# Charger et préparer les données
def load_and_prepare_data(file_path, target_column):
    df_cleaned = preparation.load_and_clean_data(file_path, "DeltaViscosity")
    df_descriptors_cleaned = preparation.generate_and_clean_descriptors(df_cleaned)
    df_filtered = preparation.filter_data_viscosity(df_descriptors_cleaned)
    return df_filtered


# Préparer les données numériques et SMILES
def prepare_numeric_and_smiles(df, target_column):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    smiles_column = 'SMILES'

    X_numeric = df[numeric_columns]
    X_smiles = df[smiles_column]
    y = np.log(df[target_column])

    return X_numeric, X_smiles, y


# Tokenizer and model from transformers
def get_tokenizer_and_model():
    tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    transformer_model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    return tokenizer, transformer_model


# Tokenize the SMILES strings
def tokenize_smiles(tokenizer, smiles_list):
    return tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")


# Normalize numerical data
def normalize_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Sauvegarder le scaler
    joblib.dump(scaler, 'scaler.pkl')

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# Split data into train, validation, and test sets
def split_data(X_numeric, X_smiles, y, test_size=0.2, random_state=42):
    X_train_numeric, X_temp_numeric, X_train_smiles, X_temp_smiles, y_train, y_temp = train_test_split(
        X_numeric, X_smiles, y, test_size=test_size, random_state=random_state)
    X_val_numeric, X_test_numeric, X_val_smiles, X_test_smiles, y_val, y_test = train_test_split(
        X_temp_numeric, X_temp_smiles, y_temp, test_size=0.5, random_state=random_state)

    return X_train_numeric, X_val_numeric, X_test_numeric, X_train_smiles, X_val_smiles, X_test_smiles, y_train, y_val, y_test


# Class Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Tanh()
        )
        self.decoder = nn.Linear(encoding_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(data, input_dim, encoding_dim=2, num_epochs=100, learning_rate=0.001):
    autoencoder = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    data_tensor = torch.tensor(data, dtype=torch.float32)

    for epoch in range(num_epochs):
        autoencoder.train()
        optimizer.zero_grad()
        reconstructions = autoencoder(data_tensor)
        loss = criterion(reconstructions, data_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    return autoencoder


def detect_outliers(autoencoder, data, threshold_percentile=90):
    autoencoder.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        reconstructions = autoencoder(data_tensor)
        mse = torch.mean((data_tensor - reconstructions) ** 2, dim=1)
        threshold = np.percentile(mse.numpy(), threshold_percentile)
        outliers = mse > threshold
    return outliers


# Combined Model with PyTorch
class CombinedModel(nn.Module):
    def __init__(self, transformer_model, numeric_input_dim):
        super(CombinedModel, self).__init__()
        self.transformer_model = transformer_model
        self.transformer_model.requires_grad_(False)  # Freeze transformer weights
        self.fc1 = nn.Linear(numeric_input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(transformer_model.config.hidden_size + 32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, input_ids, attention_mask, numeric_data):
        transformer_output = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)
        transformer_pooled_output = transformer_output.last_hidden_state.mean(dim=1)
        x = torch.relu(self.fc1(numeric_data))
        x = torch.relu(self.fc2(x))
        combined = torch.cat((transformer_pooled_output, x), dim=1)
        z = torch.relu(self.fc3(combined))
        z = torch.relu(self.fc4(z))
        return self.output(z)


def train_combined_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for input_ids, attention_mask, numeric_data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, numeric_data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for input_ids, attention_mask, numeric_data, targets in val_loader:
                outputs = model(input_ids, attention_mask, numeric_data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


def evaluate_combined_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, numeric_data, targets in test_loader:
            outputs = model(input_ids, attention_mask, numeric_data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Mean Squared Error on test set: {test_loss:.4f}")


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# Main script
if __name__ == "__main__":
    file_path = '/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/data/IlThermo-smiled_dataset_viscosity.csv'
    target_column = 'Viscosity Pa/s'

    # Charger et préparer les données
    df = load_and_prepare_data(file_path, target_column)
    print(df.columns)
    X_numeric, X_smiles, y = prepare_numeric_and_smiles(df, target_column)

    # Split des données
    X_train_numeric, X_val_numeric, X_test_numeric, X_train_smiles, X_val_smiles, X_test_smiles, y_train, y_val, y_test = split_data(
        X_numeric, X_smiles, y)

    # Tokenizer and transformer model
    tokenizer, transformer_model = get_tokenizer_and_model()

    # Tokenize SMILES
    X_train_smiles_tokenized = tokenize_smiles(tokenizer, X_train_smiles.tolist())
    X_val_smiles_tokenized = tokenize_smiles(tokenizer, X_val_smiles.tolist())
    X_test_smiles_tokenized = tokenize_smiles(tokenizer, X_test_smiles.tolist())

    # Normalize numeric data
    X_train_numeric_scaled, X_val_numeric_scaled, X_test_numeric_scaled, scaler = normalize_data(X_train_numeric,
                                                                                                 X_val_numeric,
                                                                                                 X_test_numeric)

    # Convert numeric data to tensors
    X_train_numeric_tensor = torch.tensor(X_train_numeric_scaled, dtype=torch.float32)
    X_val_numeric_tensor = torch.tensor(X_val_numeric_scaled, dtype=torch.float32)
    X_test_numeric_tensor = torch.tensor(X_test_numeric_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_smiles_tokenized['input_ids'], X_train_smiles_tokenized['attention_mask'],
                                  X_train_numeric_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_smiles_tokenized['input_ids'], X_val_smiles_tokenized['attention_mask'],
                                X_val_numeric_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_smiles_tokenized['input_ids'], X_test_smiles_tokenized['attention_mask'],
                                 X_test_numeric_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize and train the combined model
    model = CombinedModel(transformer_model, X_train_numeric_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_combined_model(model, train_loader, val_loader, criterion, optimizer, epochs=8)

    # Evaluate the model
    evaluate_combined_model(model, test_loader, criterion)

    # Save the model
    model_path = 'combined_model.pth'
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
