# Author:       Emma Gillespie
# Date:         2024-02-26
# Description:  A program that will tokenize words for a neural network

import numpy
import time
import regex as re

#text = 'A mythic and emotionally charged hero\'s journey, "Dune" tells the story of Paul Atreides, a brilliant and gifted young man born into a great destiny beyond his understanding, who must travel to the most dangerous planet in the universe to ensure the future of his family and his people.'

text = '''
Title: "Navigating the Cybersecurity Landscape: Protecting Your Digital Fortress"

Introduction:
In an era dominated by digital advancements, the importance of cybersecurity cannot be overstated. As technology evolves, so do the threats that seek to exploit vulnerabilities in our digital landscape. In this blog post, we'll explore the ever-changing cybersecurity landscape and discuss practical strategies to safeguard your digital fortress.

Understanding the Threat Landscape:

1. The Rise of Cyber Threats:
   The digital age has brought about unprecedented connectivity and convenience, but it has also given rise to an array of cyber threats. From ransomware attacks and phishing scams to data breaches, businesses and individuals alike are constantly under siege.

2. Targeted Attacks:
   Cybercriminals are becoming increasingly sophisticated, employing advanced techniques to target specific individuals, organizations, or industries. Understanding the motives behind these attacks is crucial for developing effective defense strategies.

Building a Robust Cybersecurity Foundation:

1. Risk Assessment:
   Begin by conducting a comprehensive risk assessment to identify potential vulnerabilities in your systems. This includes assessing the value of your data, understanding potential attack vectors, and evaluating the effectiveness of current security measures.

2. Implementing Strong Authentication:
   Strengthen your defenses by implementing multi-factor authentication (MFA) across all systems. This adds an extra layer of protection, making it significantly more challenging for unauthorized users to gain access.

3. Regular Software Updates:
   Keep all software and systems up to date with the latest security patches. Regular updates help to address known vulnerabilities and ensure that your defenses are as robust as possible.

4. Employee Training:
   Human error remains one of the leading causes of cybersecurity incidents. Provide regular training sessions to educate employees about potential threats, the importance of strong passwords, and how to recognize phishing attempts.

Advanced Cybersecurity Measures:

1. Endpoint Security:
   Invest in robust endpoint protection to secure devices connected to your network. This includes antivirus software, firewalls, and intrusion detection systems to detect and mitigate potential threats.

2. Network Security:
   Implement network security measures such as firewalls, intrusion prevention systems, and virtual private networks (VPNs) to safeguard the flow of information within your organization.

3. Data Encryption:
   Encrypt sensitive data both in transit and at rest. This ensures that even if unauthorized access occurs, the intercepted data remains unreadable without the proper decryption keys.

Conclusion:
As technology advances, so do the threats to our digital security. It is imperative for individuals and organizations to stay ahead of the curve by continuously adapting and enhancing their cybersecurity measures. By implementing a multi-faceted approach, including risk assessment, employee training, and advanced security measures, we can create a robust defense against the ever-evolving cyber threats, securing our digital future. Remember, in the realm of cybersecurity, proactive measures are the key to success.
'''

tokens  = text.encode('utf-8')
tokens = list(map(int, tokens))

print('-----')
print(f'Print Original Text:\n{text}')
print(f'length: {len(text)}')
print('-----')
print(f'Original Tokens:\n{tokens}')
print(f'length: {len(tokens)}')

def get_stats(ids):
    counts = {}

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts

stats = get_stats(tokens)
print('-----')
print(f'Get number of all pairs:\n{sorted(((v,k) for k, v in stats.items()), reverse=True)}')
print(f'length: {len(stats)}')

print('-----')
top_pair = max(stats, key=stats.get)
print(f'What is the most frequent pair:\n{top_pair} = {chr(top_pair[0])}, {chr(top_pair[1])}')

def merge(ids, pair, idx):
    newids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    
    return newids

print('-----')
tokens2 = merge(tokens, top_pair, 256)
print(tokens2)
print(f'length: {len(tokens2)}')

# Merging Multiple reoccuring characters
print('-----')

vocab_size = 276 # desired final vocab size
num_merges = vocab_size - 256
ids = list(tokens)

merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f'merging {pair} into a new token {idx}')
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print('-----')
print(f'Token Length: {len(tokens)}')
print(f'ids length: {len(ids)}')
print(f'Compression ratio: {len(tokens) / len(ids):.2f}X')

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors='replace')

    return text

print(f'-----')
print(f'Decoded:\n{decode(ids)}')

def encode(text):
    tokens = list(text.encode("utf-8"))

    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))

        if pair not in merges:
            break # Nothing else can be done

        idx = merges[pair]
        tokens = merge(tokens, pair, idx)

    return tokens

print("-----")
print(f'Encode:\n{encode(text)}')
print('-----')

gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

print('-----')
print(f'{re.findall(gpt2pat, text)}')
print('-----')