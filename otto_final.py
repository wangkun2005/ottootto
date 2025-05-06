import json
from collections import defaultdict
import pandas as pd

# 是否使用 sample（True：调试模式，仅加载前 100000 条；False：加载全部数据）
USE_SAMPLE = True
SAMPLE_SIZE = 100000

train_sessions = []
test_sessions = []
aid_set = set()

if USE_SAMPLE:
    print(f"使用 sample 模式，仅加载前 {SAMPLE_SIZE} 条训练数据")
    chunks = pd.read_json(r'X:/otto-recommender-system/train.jsonl', lines=True, chunksize=SAMPLE_SIZE)
    for chunk in chunks:
        for _, row in chunk.iterrows():
            session_id = row['session']
            events = row['events']
            events.sort(key=lambda x: x['ts'])
            aids = [e['aid'] for e in events]
            types = [e['type'] for e in events]
            train_sessions.append((session_id, aids, types))
            aid_set.update(aids)
        break  #  只加载第一个 chunk
else:
    print("使用全量训练数据")
    with open(r'X:/otto-recommender-system/train.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            session_id = data['session']
            events = data['events']
            events.sort(key=lambda x: x['ts'])
            aids = [e['aid'] for e in events]
            types = [e['type'] for e in events]
            train_sessions.append((session_id, aids, types))
            aid_set.update(aids)

# 无论是否 sample，测试集都加载全量
with open(r'X:/otto-recommender-system/test.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        session_id = data['session']
        events = data['events']
        events.sort(key=lambda x: x['ts'])
        aids = [e['aid'] for e in events]
        types = [e['type'] for e in events]
        test_sessions.append((session_id, aids, types))
        aid_set.update(aids)

# 构建商品ID索引映射
aid_list = sorted(list(aid_set))
aid2idx = {aid: idx+1 for idx, aid in enumerate(aid_list)}
idx2aid = {idx+1: aid for idx, aid in enumerate(aid_list)}
num_items = len(aid_list) + 1


# 将行为类型转换为索引
type2idx = {'clicks': 0, 'carts': 1, 'orders': 2}

# 构建训练样本列表
train_data = []
for (session_id, aids, types) in train_sessions:
    # 转为索引表示
    seq = [aid2idx[a] for a in aids]
    t_seq = [type2idx[t] for t in types]
    # 每个位置 i（从1到len-1）产生一个训练样本：前缀 seq[0:i] -> 下一个 item seq[i], 及事件类型 t_seq[i]
    for i in range(1, len(seq)):
        prefix = seq[:i]
        target_aid = seq[i]        # 下一步商品索引（正样本）
        target_type = t_seq[i]     # 事件类型任务索引
        train_data.append((prefix, target_aid, target_type))
# 简单打乱（可选）
import random
random.shuffle(train_data)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRec(nn.Module):
    def __init__(self, num_items, embed_dim=64, num_heads=2, num_layers=2, dropout=0.2):
        super(SASRec, self).__init__()
        self.embed_dim = embed_dim
        # 商品嵌入和位置嵌入
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(1000, embed_dim)  # 假设最大序列长度不超过1000
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 多任务头：点击/加购/下单的 gating 向量
        self.g_click = nn.Parameter(torch.ones(embed_dim))
        self.g_cart = nn.Parameter(torch.ones(embed_dim))
        self.g_order = nn.Parameter(torch.ones(embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, key_padding_mask=None):
        """
        seq: tensor of shape (batch, seq_len), 包含商品索引（填充位置为0）
        key_padding_mask: (batch, seq_len) bool tensor, True 表示该位置是填充需屏蔽
        返回：最后一个位置的上下文向量 (batch, embed_dim)
        """
        # 商品嵌入 + 位置嵌入
        emb = self.item_embedding(seq)  # (batch, seq_len, embed_dim)
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        h = emb + pos_emb
        h = self.dropout(h)
        # Transformer 编码（使用 key_padding_mask 屏蔽填充位置）
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        # 取每序列最后一个有效位置的输出作为上下文
        # lengths = total length excluding padding = ~
        # 由于 Transformer 不自动忽略填充，我们从 key_padding_mask 得知实际长度
        # 取最后一个非填充位置
        batch_size, seq_len, _ = h.size()
        last_h = []
        if key_padding_mask is not None:
            for i in range(batch_size):
                # 找出第 i 个序列有多少个实际条目
                valid_len = (key_padding_mask[i] == 0).sum().item()
                # 取最后一个非填充位置的输出
                if valid_len > 0:
                    last_h.append(h[i, valid_len-1, :])
                else:
                    # 极端情况：全填充，取全零向量
                    last_h.append(torch.zeros(self.embed_dim, device=h.device))
            last_h = torch.stack(last_h, dim=0)  # (batch, embed_dim)
        else:
            # 如果无填充（batch内等长），则直接取最后一列
            last_h = h[:, -1, :]  # (batch, embed_dim)
        return last_h  # (batch, embed_dim)

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 自定义 Dataset
class OttoDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list  # list of (prefix_seq, target_aid, target_type)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        prefix, aid, ttype = self.data[idx]
        return prefix, aid, ttype

# 生成 DataLoader 时需要 Pad 填充
def collate_batch(batch):
    sequences, pos_items, types = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    padded_seqs = []
    key_padding_mask = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded_seqs.append(seq + [0]*pad_len)  # 填充0
        key_padding_mask.append([0]*len(seq) + [1]*pad_len)  # 1 表示填充位置
    return (torch.LongTensor(padded_seqs),
            torch.BoolTensor(key_padding_mask),
            torch.LongTensor(pos_items),
            torch.LongTensor(types))

# 划分训练/验证集（例如80%训练，20%验证）
random.shuffle(train_data)
split = int(0.8 * len(train_data))
train_list = train_data[:split]
val_list = train_data[split:]

train_loader = DataLoader(OttoDataset(train_list), batch_size=32,
                          shuffle=True, collate_fn=collate_batch)
val_loader   = DataLoader(OttoDataset(val_list), batch_size=32,
                          shuffle=False, collate_fn=collate_batch)

# 初始化模型与优化器
model = SASRec(num_items=num_items, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# 负样本数量
n_neg = 10

# 训练循环（简化，未包括早停等细节）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

import time
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(5):
    print(f"\n--- Epoch {epoch+1}/5 开始 ---")
    start_time = time.time()

    model.train()
    total_loss = 0.0
    batch_count = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        seqs, padding_mask, pos_items, types = batch
        seqs = seqs.to(device)
        padding_mask = padding_mask.to(device)
        pos_items = pos_items.to(device)
        types = types.to(device)

        optimizer.zero_grad()
        context = model(seqs, key_padding_mask=padding_mask)

        batch_size = context.size(0)
        embed_dim = context.size(1)

        g_vectors = torch.stack([model.g_click, model.g_cart, model.g_order])[types]
        h_t = context * g_vectors
        pos_emb = model.item_embedding(pos_items)
        logit_pos = torch.sum(h_t * pos_emb, dim=1)
        loss_pos = F.softplus(-logit_pos)

        neg_ids = torch.randint(1, num_items, (batch_size, n_neg), device=device)
        neg_embs = model.item_embedding(neg_ids)
        h_t_expand = h_t.unsqueeze(1)
        logits_neg = torch.sum(h_t_expand * neg_embs, dim=2)
        loss_neg = F.softplus(logits_neg).sum(dim=1)

        loss = loss_pos + loss_neg
        batch_loss = loss.mean()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

        # 🟡 日志：每 100 个 batch 输出一次
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == batch_count:
            elapsed = time.time() - start_time
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                  f"Epoch {epoch+1} | Batch {batch_idx+1}/{batch_count} | "
                  f"AvgLoss: {total_loss/(batch_idx+1):.4f} | Time Elapsed: {elapsed:.1f}s")

    print(f"✅ Epoch {epoch+1} 结束，总训练损失：{total_loss:.4f}，耗时：{(time.time() - start_time):.1f}s")
    
    
    # --- 验证阶段（简化示意） ---
    model.eval()
    all_labels = {0: [], 1: [], 2: []}
    all_scores = {0: [], 1: [], 2: []}
    hit_count = {0: 0, 1: 0, 2: 0}
    total_count = {0: 0, 1: 0, 2: 0}
    with torch.no_grad():
        for batch in val_loader:
            seqs, padding_mask, pos_items, types = batch
            seqs, padding_mask, pos_items, types = seqs.cuda(), padding_mask.cuda(), pos_items.cuda(), types.cuda()
            context = model(seqs, key_padding_mask=padding_mask)
            for i in range(context.size(0)):
                h = context[i]; t = types[i].item(); pos = pos_items[i].item()
                total_count[t] += 1
                # 计算该样本对应任务的分数（对正负样本分别记录）
                if t == 0: g = model.g_click
                elif t == 1: g = model.g_cart
                else: g = model.g_order
                h_t = h * g
                score_pos = torch.dot(h_t, model.item_embedding.weight[pos]).item()
                all_scores[t].append(score_pos)
                all_labels[t].append(1)  # 正样本标记1
                # 随机选几个负样本进行 AUC 评估
                neg_ids = torch.randint(1, num_items, (n_neg,), device=h.device)
                for neg in neg_ids:
                    score_neg = torch.dot(h_t, model.item_embedding.weight[neg]).item()
                    all_scores[t].append(score_neg)
                    all_labels[t].append(0)
                # 计算 Hit Rate@20：在这里简化为检查正样本是否在 top-20（此处仅示意）
                # 实际应用中可对所有商品评分并取topK，此处不逐一实现。
                # 假设有一个函数 check_hit(top20_list, pos) 判断命中。这里略写：
                # top20_list = ... 
                # if pos in top20_list: hit_count[t] += 1
                # ...
        # 计算各任务 AUC（使用 sklearn 或其他库；此处不展开具体实现）
        from sklearn.metrics import roc_auc_score
        aucs = {}
        for t in [0,1,2]:
            if any(all_labels[t]):
                auc = roc_auc_score(all_labels[t], all_scores[t])
            else:
                auc = float('nan')
            aucs[t] = auc
        # --- 计算 Hit Rate@20 ---
    # 获取所有商品的嵌入（排除 padding_idx=0）
        item_embs = model.item_embedding.weight  # (num_items, embed_dim)

        for i in range(context.size(0)):
           h = context[i]             # 当前样本上下文向量
           t = types[i].item()        # 当前样本的行为类型
           pos = pos_items[i].item()  # 正样本 aid 索引

           total_count[t] += 1
           if t == 0: g = model.g_click
           elif t == 1: g = model.g_cart
           else: g = model.g_order
           h_t = h * g  # 加权上下文向量

    # 与所有商品计算打分
           scores = torch.matmul(item_embs, h_t)  # (num_items,)

    # 排除 padding（索引为0）
           scores[0] = -float('inf')

    # 取 top-20 物品索引
           topk = torch.topk(scores, 20).indices.tolist()

           if pos in topk:
               hit_count[t] += 1

        print(f"Val AUC: click={aucs[0]:.4f}, cart={aucs[1]:.4f}, order={aucs[2]:.4f}")
        print(f"🎯 Val HitRate@20: click={hit_rate[0]:.4f}, cart={hit_rate[1]:.4f}, order={hit_rate[2]:.4f}")

import pandas as pd

model.eval()
submission_rows = []
with torch.no_grad():
    # 分批处理测试会话以加速（同样要填充序列）
    test_loader = DataLoader(OttoDataset([(None, aids, types) for (sid, aids, types) in test_sessions]),
                             batch_size=32, collate_fn=collate_batch)
    idx = 0
    for batch in test_loader:
        seqs, padding_mask, _, _ = batch  # 测试集无正负标签，这里 pos_items 和 types 都可置为 0 占位
        seqs, padding_mask = seqs.cuda(), padding_mask.cuda()
        context = model(seqs, key_padding_mask=padding_mask)  # (batch, embed_dim)
        batch_size = context.size(0)
        for i in range(batch_size):
            sid, aids, types_seq = test_sessions[idx]
            idx += 1
            h = context[i]  # 上下文向量
            # 对每种行为类型分别计算分数并选 top20
            recs = {}
            for t, type_name, g in [(0, 'clicks', model.g_click),
                                     (1, 'carts', model.g_cart),
                                     (2, 'orders', model.g_order)]:
                h_t = h * g
                # 计算所有商品的分数（此处简化，实际可优化）
                scores = torch.matmul(model.item_embedding.weight, h_t)  # (num_items,)
                # 排除索引0和当前会话已有的商品
                scores[0] = -1e9
                seen = set(aids)
                for aid in seen:
                    if aid in aid2idx:
                        scores[aid2idx[aid]] = -1e9
                topk = torch.topk(scores, 20).indices.cpu().tolist()
                rec_items = [idx2aid[idx] for idx in topk]
                recs[type_name] = rec_items
            # 添加到提交行
            session_str = str(sid)
            for t in ['clicks', 'carts', 'orders']:
                session_type = session_str + "_" + t
                pred_str = " ".join(str(x) for x in recs[t])
                submission_rows.append((session_type, pred_str))

# 保存为 CSV
sub_df = pd.DataFrame(submission_rows, columns=['session_type', 'prediction'])
sub_df.to_csv('submission.csv', index=False)

