from modelDesign_1 import build_model


def Model_2(input_shape, output_shape):
    model = build_model(input_shape,
                        output_shape,
                        embed_dim=256,
                        hidden_dim=512,
                        num_heads=8,
                        num_attention_layers=6)
    return model


if __name__ == '__main__':
    model = Model_2((72, 2, 256), 2)
    model.summary()
