import tensorflow as tf


class DataAugmentation(tf.keras.Model):
    def __init__(self, img_size):
        super(DataAugmentation, self).__init__()
        self.norm = tf.keras.layers.Normalization()
        self.res = tf.keras.layers.Resizing(img_size, img_size)
        self.flip = tf.keras.layers.RandomFlip("horizontal")
        self.rotation = tf.keras.layers.RandomRotation(0.03)

    def call(self, inputs):
        x = self.res(inputs)
        x = self.flip(x)
        x = self.rotation(x)
        return x
# Class that turns images into patches


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        # Define how big the patches are
        self.patch_size = patch_size

    def call(self, inputs):
        # Find the batch size
        batch_size = tf.shape(inputs)[0]
        # Extract patch_size by patch_size patches from images
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # Reshape the pathces into batch_size by patch_size by patch_size
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Class that turns patches into a linear representation and adds a learnable positional embedding


class PatchEmbeddings(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEmbeddings, self).__init__()
        # Get the number of patchese
        self.num_patches = num_patches
        # Create a projection layer
        self.projection = tf.keras.layers.Dense(projection_dim)
        # Create a learnable embedding layer
        self.positional_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, inputs):
        # Create the positions
        positions = tf.range(0, self.num_patches, delta=1)
        # Apply the projections to the patches and apply the positional embeddings then add
        encoded = self.projection(
            inputs) + self.positional_embedding(positions)
        return encoded

# Class that creates a simple MLP for the transformer


class MLP(tf.keras.layers.Layer):
    def __init__(self, units):
        super(MLP, self).__init__()
        # First Layer
        self.layer1 = tf.keras.layers.Dense(units[0])
        # Gaussian Error Linear Unit Activation
        # Note: This activation is very similar to ReLU and is the
        # preferred activation in Transformer-based models, like ViT or BERT
        self.gelu = tf.keras.layers.Activation(tf.nn.gelu)
        # Output Layer
        self.outputs = tf.keras.layers.Dense(units[1])

    def call(self, inputs):
        # Call the layers
        return self.outputs(self.gelu(self.layer1(inputs)))

# Class for the classification head at the top of the ViT


class ClassificationHead(tf.keras.Model):
    def __init__(self, dropout_rate, num_classes=100):
        super(ClassificationHead, self).__init__()
        self.drop = tf.keras.layers.Dropout(dropout_rate)
        self.outputs = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.drop(inputs)
        return self.outputs(x)


class TransformerBlock(tf.keras.Model):
    def __init__(self, dim, mlp_dim, attention_heads):
        super(TransformerBlock, self).__init__()
        # # Create attention layer
        # Layer normalization 1.
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        # Create a multi-head attention layer.
        self.attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=attention_heads, key_dim=dim, dropout=0.1
        )
        self.add1 = tf.keras.layers.Add()
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.mlp = MLP(mlp_dim)
        self.add2 = tf.keras.layers.Add()

    def call(self, inputs):
        # Call transformer block
        x1 = self.ln1(inputs)
        attention = self.attention_output(x1, x1)
        x2 = self.add1([attention, inputs])
        x3 = self.ln2(x2)
        x3 = self.mlp(x3)
        x = self.add2([x3, x2])
        return x

# Create a Transformer


class Transformer(tf.keras.Model):
    def __init__(self, dim, mlp_dim, attention_heads, depth):
        super(Transformer, self).__init__()
        # Create a list of `depth` transformers
        transformer_list = []
        for _ in range(depth):
            transformer_list.append(TransformerBlock(
                dim, mlp_dim, attention_heads))
        self.net = tf.keras.Sequential(transformer_list)

    def call(self, inputs):
        x = self.net(inputs)
        return x

# Build a Vision Transformer


class ViT(tf.keras.Model):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,
                 dim,
                 depth,
                 attention_heads,
                 transformer_units,
                 mlp_head_units,
                 augment=False
                 ):
        super(ViT, self).__init__()
        # Initialize needed variables and calculate num_patches
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.projection_dim = dim
        self.transformer_layers = depth
        self.attention_heads = attention_heads
        self.mlp_units = transformer_units
        self.augment = augment
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Initialize layers
        # self.aug = DataAugmentation(self.image_size)
        self.patches = Patches(self.patch_size)
        self.patchreprs = PatchEmbeddings(
            self.num_patches, self.projection_dim)
        # Build a transformer
        self.transformer = Transformer(
            self.projection_dim, self.mlp_units, self.attention_heads, self.transformer_layers)
        # Create Linear Representation
        self.linear_repr = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon=1e-5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3)
        ])
        self.mlp = MLP(mlp_head_units)
        # Initialize a classification head
        self.classifier = ClassificationHead(0.3)

    def call(self, input):
        if augment:
            x = self.aug(input)
            patches = self.patches(x)
        else:
            patches = self.patches(input)
        x = self.patchreprs(patches)
        x = self.transformer(x)
        x = self.linear_repr(x)
        x = self.mlp(x)
        x = self.classifier(x)
        return x
