"""Prove whole-prefix loop keeps tile order and every removed mask is true."""
count = 0
for capacity in (17, 65, 513, 2084):
    for width in (16, 32, 64, 128):
        for splits in (1, 2, 5, 18):
            chunk = (capacity + splits - 1) // splits
            for position in sorted({0, 1, min(width - 1, capacity-1), capacity//2, capacity-1}):
                for tokens in (1, 2, 4, 16):
                    if position + tokens > capacity:
                        continue
                    for split in range(splits):
                        begin = split * chunk
                        first = position + 1
                        end = min(begin+chunk, capacity, first+tokens-1)
                        whole = begin + max(min(end, first)-begin, 0)//width*width
                        assert list(range(begin, whole, width)) + list(range(whole, end, width)) == list(range(begin, end, width))
                        for start in range(begin, whole, width):
                            for j in range(start, start+width):
                                assert 0 <= j < end <= capacity
                                for chain in (1, tokens):
                                    for t in range(tokens):
                                        valid = first + (t if t < chain else 0)
                                        own = -1 if t < chain else first - 1 + t
                                        assert j < valid or j == own
                        count += 1
print(f'{count} split/tile/tree boundary configurations preserve order and masks')
