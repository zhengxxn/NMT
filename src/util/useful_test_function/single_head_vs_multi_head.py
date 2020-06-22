import torch
import torch.nn as nn

head_num = 4
head_dim = 2
batch_size = 3
seq_len = 5
global_memory_count = 10

token_inp = torch.randn(batch_size, seq_len, head_num * head_dim)
global_key = nn.Linear(head_num * head_dim, global_memory_count, bias=False)
global_value = nn.Linear(head_num * head_dim, global_memory_count, bias=False)
global_key_vector = global_key.weight   # [10, 8]
global_value_vector = global_value.weight   # [10, 8]
print(global_key_vector.shape)

# single head attention
score = torch.matmul(token_inp, global_key_vector.t())  # [batch, seq, memory count]
score = torch.relu(score)
single_head_output = torch.matmul(score, global_value_vector)  # [batch, seq, hid dim]

# multi head attention
query = token_inp.view(batch_size, -1, head_num, head_dim).transpose(1, 2)  # [b, h_n, s, h_d]
key = global_key_vector.view(global_memory_count, head_num, head_dim).transpose(0, 1)  # [h_n, m_n, h_d]

score = torch.matmul(query, key.transpose(-1, -2))  # [b, h_n, s, m_n]
score2 = torch.einsum('bij,abcj->abci', (key, query))
print(score.shape)
print(score2.shape)
print(torch.equal(score, score2))

score = torch.relu(score)
score2 = torch.relu(score2)

value = global_value_vector.view(global_memory_count, head_num, head_dim).transpose(0, 1)
multi_head_output = torch.matmul(score, value)  # [b, h_n, s, h_d]
multi_head_output2 = torch.einsum('bij,abci->abcj', (value, score2))

print(multi_head_output.shape)
print(multi_head_output2.shape)

print(torch.equal(multi_head_output, multi_head_output2))

multi_head_output = multi_head_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
loss = torch.sum(multi_head_output)


loss.backward()

# print(single_head_output)
# print(multi_head_output)
print(global_key.weight.grad)
# loss = torch.matmul(token_inp, global_key_vector.t())
# # loss = global_key(token_inp)
# loss = torch.sum(loss)
# print(loss)
# loss.backward()
# print(global_key.weight.grad)
