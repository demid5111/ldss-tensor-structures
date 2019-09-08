"""
    We have two sentences:
    1. Few leaders are admired by George (Passive voice)
    2. George admires few leaders (Active voice)

    When using the notation from the original paper
    Legendre, G., Miyata, Y., & Smolensky, P. (1991).
    Distributed recursive structure processing.
    In Advances in Neural Information Processing Systems (pp. 591-597).
    Those trees are:

    Active voice
      (root)
    /         \
    A       /     \
          V       P

    Passive voice
      (root)
    /         \
    P       /           \
          /     \      /       \
        Aux     V     by        A


    We need to extract the following structure:

      (root)
    /         \
    V       /  \
          A     P
"""


if __name__ == '__main__':
    print('Hello, Active-Passive Net')



    pass
