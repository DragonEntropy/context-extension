from ALiBi import LlamaAttentionALiBi


def test():
    num_heads = 4
    slopes = LlamaAttentionALiBi.precompute_slopes(num_heads)
    alibi = LlamaAttentionALiBi.calculate_alibi(slopes, 10, 10)
    print(alibi)


if __name__ == "__main__":
    test()
