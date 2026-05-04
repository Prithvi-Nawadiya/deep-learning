import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Parameters
input_dim = 20
latent_dim = 2

# ---------------- ENCODER ----------------
encoder_inputs = layers.Input(shape=(input_dim,))
x = layers.Dense(10, activation='relu')(encoder_inputs)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])

encoder = Model(encoder_inputs, [z_mean, z_log_var, z])

# ---------------- DECODER ----------------
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(10, activation='relu')(latent_inputs)
decoder_outputs = layers.Dense(input_dim, activation='sigmoid')(x)

decoder = Model(latent_inputs, decoder_outputs)

# ---------------- VAE MODEL ----------------
class VAE(Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

vae = VAE(encoder, decoder)
vae.compile(optimizer='adam')

x_train = np.random.rand(1000, input_dim).astype("float32")

vae.fit(x_train, epochs=20, batch_size=32)

z_sample = np.random.normal(size=(10, latent_dim)).astype("float32")
generated = decoder.predict(z_sample)

print("Generated Data:\n", generated)


# ---------------- GAN ----------------
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(10,)),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(1,)),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False

gan_input = layers.Input(shape=(10,))
fake = generator(gan_input)
gan_output = discriminator(fake)

gan = Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

epochs = 1000
batch_size = 32

for epoch in range(epochs):

    real_data = np.random.rand(batch_size, 1)
    real_labels = np.ones((batch_size, 1))

    noise = np.random.rand(batch_size, 10)
    fake_data = generator.predict(noise, verbose=0)
    fake_labels = np.zeros((batch_size, 1))

    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_data, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

    discriminator.trainable = False
    noise = np.random.rand(batch_size, 10)
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss_real[0]+d_loss_fake[0]}, G Loss: {g_loss}")

noise = np.random.rand(5, 10)
generated = generator.predict(noise)
print("Generated Data:\n", generated)


# ---------------- GCN ----------------
import torch
import torch.nn as nn
import torch.optim as optim

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = torch.relu(self.fc1(x))
        x = torch.matmul(adj, x)
        x = self.fc2(x)
        return x

num_nodes = 6
features = torch.rand(num_nodes, 10)

adj = torch.eye(num_nodes)

labels = torch.randint(0, 2, (num_nodes,))

model = GCN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(50):
    optimizer.zero_grad()

    output = model(features, adj)
    loss = loss_fn(output, labels)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

pred = torch.argmax(model(features, adj), dim=1)
print("Predicted Classes:", pred)
