class settings:

    train_size = 0.8
    block_size = 256
    batch_size = 64
    embedding_size = 384

    max_new_tokens = 100
    head_size = 384//4 # head_size // heads
    n_heads = 4 # heads //
    n_decoder_blocks = 6
    dropout = 0.2
    lr = 3e-4