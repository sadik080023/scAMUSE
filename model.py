"""
model.py

scAMUSE - Enhanced DCAM with Validation ARI Early Stopping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# ============================================================
# Adaptive Weight Scheduler
# ============================================================

class AdaptiveWeightScheduler:
    """Phase-wise loss weight scheduling."""
    
    def __init__(self, total_epochs, warmup_epochs=30, ramp_epochs=40,
                 target_w_dcam=0.1, target_w_energy=0.05,
                 target_w_entropy=0.2, target_w_diversity=0.5,
                 target_w_util=0.1, target_w_attraction=1.0):
        
        self.total_epochs = total_epochs
        self.warmup_epochs = min(warmup_epochs, total_epochs)
        self.ramp_epochs = min(ramp_epochs, total_epochs - self.warmup_epochs)
        self.fine_tune_start = self.warmup_epochs + self.ramp_epochs
        
        self.target_w_dcam = target_w_dcam
        self.target_w_energy = target_w_energy
        self.target_w_entropy = target_w_entropy
        self.target_w_diversity = target_w_diversity
        self.target_w_util = target_w_util
        self.target_w_attraction = target_w_attraction
    
    def get_weights(self, epoch):
        """Return weights based on current epoch (0-indexed)."""
        
        if epoch < self.warmup_epochs:
            # Phase 1: Warmup - gradually increase from 10% to 50% of target
            progress = epoch / max(self.warmup_epochs, 1)
            w_dcam = self.target_w_dcam * (0.1 + 0.4 * progress)  # 0.01 → 0.05
            w_energy = self.target_w_energy * 0.3 * progress
            w_entropy = self.target_w_entropy * 0.3 * progress
            w_diversity = self.target_w_diversity * 0.3 * progress
            w_util = self.target_w_util * 0.3 * progress
            w_attraction = self.target_w_attraction * 0.3 * progress
            
        elif epoch < self.fine_tune_start:
            # Phase 2: Ramp - gradually increase to full target
            progress = (epoch - self.warmup_epochs) / max(self.ramp_epochs, 1)
            w_dcam = self.target_w_dcam * (0.5 + 0.5 * progress)  # 0.05 → 0.1
            w_energy = self.target_w_energy * (0.3 + 0.7 * progress)
            w_entropy = self.target_w_entropy * (0.3 + 0.7 * progress)
            w_diversity = self.target_w_diversity * (0.3 + 0.7 * progress)
            w_util = self.target_w_util * (0.3 + 0.7 * progress)
            w_attraction = self.target_w_attraction * (0.3 + 0.7 * progress)
            
        else:
            # Phase 3: Fine-tune - full weights
            w_dcam = self.target_w_dcam
            w_energy = self.target_w_energy
            w_entropy = self.target_w_entropy
            w_diversity = self.target_w_diversity
            w_util = self.target_w_util
            w_attraction = self.target_w_attraction
        
        return {
            "w_dcam": w_dcam,
            "w_energy": w_energy,
            "w_entropy": w_entropy,
            "w_diversity": w_diversity,
            "w_util": w_util,
            "w_attraction": w_attraction,
        }


# ============================================================
# Supervised Loss Components
# ============================================================

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        valid_mask = labels != -1
        if valid_mask.sum() <= 1:
            return features.sum() * 0.0
        features = features[valid_mask]
        labels = labels[valid_mask]
        labels = labels.contiguous().view(-1, 1)
        n = features.shape[0]
        mask = torch.eq(labels, labels.T).float().to(device)
        logits = torch.div(features @ features.T, self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        logits_mask = torch.ones_like(mask) - torch.eye(n, device=device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        return -mean_log_prob_pos.mean()


class SemiSupervisedLoss(nn.Module):
    def __init__(self, n_classes, w_triplet=1.0, w_pairwise=0.1, w_supcon=0.5,
                 w_ce=1.0, triplet_margin=0.5, device="cpu"):
        super().__init__()
        self.n_classes = n_classes
        self.w_triplet = w_triplet
        self.w_pairwise = w_pairwise
        self.w_supcon = w_supcon
        self.w_ce = w_ce
        self.triplet_margin = triplet_margin
        self.device = device
        self.supcon = SupConLoss(temperature=0.1)

    def triplet_loss(self, embeddings, labels, mask_labeled):
        if mask_labeled.sum() < 3:
            return torch.tensor(0.0, device=embeddings.device)
        z_labeled = embeddings[mask_labeled]
        y_labeled = labels[mask_labeled]
        loss = torch.tensor(0.0, device=embeddings.device)
        n_triplets = 0
        for i in range(len(z_labeled)):
            anchor = z_labeled[i:i+1]
            anchor_label = y_labeled[i]
            pos_mask = y_labeled == anchor_label
            pos_mask[i] = False
            neg_mask = y_labeled != anchor_label
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            pos_idx = torch.where(pos_mask)[0]
            neg_idx = torch.where(neg_mask)[0]
            rand_pos = torch.randint(0, len(pos_idx), (1,), device=self.device)
            rand_neg = torch.randint(0, len(neg_idx), (1,), device=self.device)
            pos = z_labeled[pos_idx[rand_pos]]
            neg = z_labeled[neg_idx[rand_neg]]
            pos_dist = 1.0 - F.cosine_similarity(anchor, pos, dim=1)
            neg_dist = 1.0 - F.cosine_similarity(anchor, neg, dim=1)
            loss += F.relu(pos_dist - neg_dist + self.triplet_margin).mean()
            n_triplets += 1
        if n_triplets == 0:
            return torch.tensor(0.0, device=embeddings.device)
        return loss / n_triplets

    def pairwise_loss(self, embeddings, labels, mask_labeled):
        if mask_labeled.sum() < 2:
            return torch.tensor(0.0, device=embeddings.device)
        z_labeled = embeddings[mask_labeled]
        y_labeled = labels[mask_labeled]
        sim_matrix = torch.mm(z_labeled, z_labeled.T)
        label_matrix = (y_labeled.unsqueeze(0) == y_labeled.unsqueeze(1)).float()
        eye = torch.eye(label_matrix.shape[0], device=self.device)
        label_matrix = label_matrix * (1.0 - eye)
        pos_loss = (1.0 - sim_matrix) * label_matrix
        neg_loss = F.relu(sim_matrix) * (1.0 - label_matrix)
        n_pos = label_matrix.sum()
        n_neg = (1.0 - label_matrix).sum()
        return pos_loss.sum() / (n_pos + 1e-8) + neg_loss.sum() / (n_neg + 1e-8)

    def ce_loss(self, logits, labels, mask_labeled):
        if mask_labeled.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        logits_labeled = logits[mask_labeled]
        labels_labeled = labels[mask_labeled]
        return F.cross_entropy(logits_labeled, labels_labeled)

    def forward(self, embeddings, logits, labels):
        mask_labeled = labels >= 0
        l_triplet = self.triplet_loss(embeddings, labels, mask_labeled) if self.w_triplet > 0 else torch.tensor(0.0, device=embeddings.device)
        l_pairwise = self.pairwise_loss(embeddings, labels, mask_labeled) if self.w_pairwise > 0 else torch.tensor(0.0, device=embeddings.device)
        l_supcon = self.supcon(embeddings, labels) if self.w_supcon > 0 else torch.tensor(0.0, device=embeddings.device)
        l_ce = self.ce_loss(logits, labels, mask_labeled) if self.w_ce > 0 else torch.tensor(0.0, device=embeddings.device)
        l_supervised = (self.w_triplet * l_triplet + self.w_pairwise * l_pairwise +
                        self.w_supcon * l_supcon + self.w_ce * l_ce)
        return {
            "total": l_supervised, "triplet": l_triplet, "pairwise": l_pairwise,
            "supcon": l_supcon, "ce": l_ce
        }


# ============================================================
# Refinement Head
# ============================================================

class RefinementHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.projector = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)


# ============================================================
# Enhanced Fusion Trainer
# ============================================================

class EnhancedFusionTrainer:
    def __init__(self, input_dim, n_classes,
                 n_memories=None, n_anchor_slots=None,
                 memory_temperature=0.1, memory_update_strength=0.5,
                 memory_n_iters=3, hidden_dim=256, output_dim=128,
                 lr=1e-3, weight_decay=1e-5, device="cpu", seed=42,
                 w_dcam=0.1, w_dcam_energy=0.05, w_dcam_entropy=0.2,
                 w_dcam_diversity=0.5, w_dcam_util=0.1, w_dcam_attraction=1.0,
                 w_triplet=1.0, w_pairwise=0.1,
                 w_supcon=0.5, w_ce=1.0, triplet_margin=0.5,
                 init_from_labels=True, adaptive_weights=True,
                 warmup_epochs=30, ramp_epochs=40, total_epochs=100,
                 early_stop_metric="loss"):

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.n_classes = n_classes
        self.device = device
        self.init_from_labels = init_from_labels
        self.adaptive_weights = adaptive_weights
        self.early_stop_metric = early_stop_metric
        self.triplet_margin = triplet_margin

        self.target_w_dcam = w_dcam
        self.target_w_dcam_energy = w_dcam_energy
        self.target_w_dcam_entropy = w_dcam_entropy
        self.target_w_dcam_diversity = w_dcam_diversity
        self.target_w_dcam_util = w_dcam_util
        self.target_w_dcam_attraction = w_dcam_attraction

        if n_anchor_slots is None:
            n_anchor_slots = n_classes
        if n_memories is None:
            extra_slots = max(3, n_classes // 5)
            n_memories = n_classes + extra_slots
        if n_anchor_slots > n_memories:
            raise ValueError(f"n_anchor_slots ({n_anchor_slots}) > n_memories ({n_memories})")

        self.n_anchor_slots = n_anchor_slots
        self.n_memories = n_memories
        self.n_free_slots = n_memories - n_anchor_slots

        if adaptive_weights:
            self.scheduler = AdaptiveWeightScheduler(
                total_epochs=total_epochs,
                warmup_epochs=warmup_epochs,
                ramp_epochs=ramp_epochs,
                target_w_dcam=w_dcam,
                target_w_energy=w_dcam_energy,
                target_w_entropy=w_dcam_entropy,
                target_w_diversity=w_dcam_diversity,
                target_w_util=w_dcam_util,
                target_w_attraction=w_dcam_attraction
            )
        else:
            self.scheduler = None

        from dcam_memory import DCAMemory
        
        self.dcam_memory = DCAMemory(
            embedding_dim=output_dim,
            n_memories=n_memories,
            n_anchor_slots=n_anchor_slots,
            temperature=memory_temperature,
            update_strength=memory_update_strength,
            n_iters=memory_n_iters
        ).to(device)

        self.refinement_head = RefinementHead(input_dim, hidden_dim, output_dim).to(device)
        self.classifier = nn.Linear(output_dim, n_classes).to(device)
        self.criterion = SemiSupervisedLoss(
            n_classes, w_triplet=w_triplet, w_pairwise=w_pairwise,
            w_supcon=w_supcon, w_ce=w_ce,
            triplet_margin=triplet_margin, device=device)

        all_params = (
            list(self.refinement_head.parameters()) +
            list(self.dcam_memory.parameters()) +
            list(self.classifier.parameters()) +
            list(self.criterion.parameters())
        )
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
        
        # For best model saving
        self.best_state = None

    def initialize_memories_from_labels(self, embeddings, labels, idx_labeled, n_classes):
        if not self.init_from_labels:
            return
        embeddings_t = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)
        idx_t = torch.tensor(idx_labeled, dtype=torch.long, device=self.device)
        self.dcam_memory.initialize_from_labels(embeddings_t, labels_t, idx_t, n_classes)

    def _get_dcam_weights(self, epoch):
        if self.scheduler is not None:
            return self.scheduler.get_weights(epoch)
        return {
            "w_dcam": self.target_w_dcam, "w_energy": self.target_w_dcam_energy,
            "w_entropy": self.target_w_dcam_entropy, "w_diversity": self.target_w_dcam_diversity,
            "w_util": self.target_w_dcam_util, "w_attraction": self.target_w_dcam_attraction,
        }

    def fit_transform(self, embeddings, labels, val_embeddings=None, val_labels=None,
                      epochs=100, batch_size=256, patience=10, val_eval_interval=5, verbose=True):
        
        n_samples = embeddings.shape[0]
        embeddings_t = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)

        best_val_loss = float("inf")
        best_val_ari = -1.0
        patience_counter = 0

        if verbose:
            print(f"   Enhanced DCAM: {self.n_memories} slots ({self.n_anchor_slots} anchors)")
            print(f"   Adaptive weights: {self.scheduler is not None}")
            print(f"   Early stopping metric: {self.early_stop_metric}")

        self.refinement_head.train()
        self.dcam_memory.train()
        self.classifier.train()

        for epoch in range(epochs):
            perm = torch.randperm(n_samples, device=self.device)
            epoch_losses = {
                "total": 0, "supervised": 0, "dcam": 0,
                "triplet": 0, "pairwise": 0, "supcon": 0,
                "ce": 0
            }
            n_batches = 0
            all_assignments = []

            weights = self._get_dcam_weights(epoch)
            w_dcam = weights["w_dcam"]
            w_energy = weights["w_energy"]
            w_entropy = weights["w_entropy"]
            w_diversity = weights["w_diversity"]
            w_util = weights["w_util"]
            w_attraction = weights["w_attraction"]

            for i in range(0, n_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                x_batch = embeddings_t[batch_idx]
                batch_labels = labels_t[batch_idx]

                z = self.refinement_head(x_batch)
                z_memory, assignment, energy = self.dcam_memory(z)
                logits = self.classifier(z_memory)

                sup_loss_dict = self.criterion(z_memory, logits, batch_labels)
                l_supervised = sup_loss_dict["total"]

                dcam_dict = self.dcam_memory.dcam_loss(
                    z_memory, energy, assignment,
                    w_energy=w_energy, w_entropy=w_entropy,
                    w_diversity=w_diversity, w_util=w_util,
                    w_attraction=w_attraction
                )
                l_dcam = dcam_dict["dcam_total"]

                total_loss = l_supervised + w_dcam * l_dcam

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 1.0)
                self.optimizer.step()

                epoch_losses["total"] += total_loss.item()
                epoch_losses["supervised"] += l_supervised.item()
                epoch_losses["dcam"] += l_dcam.item()
                for k in ["triplet", "pairwise", "supcon", "ce"]:
                    epoch_losses[k] += sup_loss_dict.get(k, 0)
                
                n_batches += 1
                all_assignments.append(assignment.detach().cpu())

            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            full_assignment = torch.cat(all_assignments, dim=0)
            memory_stats = self.dcam_memory.get_memory_usage_stats(full_assignment)

            # Validation
            val_loss = None
            val_ari = None
            if val_embeddings is not None and val_labels is not None:
                self.refinement_head.eval()
                self.dcam_memory.eval()
                self.classifier.eval()
                
                with torch.no_grad():
                    val_emb_t = torch.tensor(val_embeddings, dtype=torch.float32, device=self.device)
                    val_labels_t = torch.tensor(val_labels, dtype=torch.long, device=self.device)
                    z_val = self.refinement_head(val_emb_t)
                    z_val_memory, _, _ = self.dcam_memory(z_val)
                    logits_val = self.classifier(z_val_memory)
                    val_loss = F.cross_entropy(logits_val, val_labels_t).item()
                    
                    need_ari = (self.early_stop_metric == "ari") or ((epoch + 1) % val_eval_interval == 0)
                    if need_ari:
                        z_val_np = z_val_memory.cpu().numpy()
                        kmeans_val = KMeans(n_clusters=self.n_classes, random_state=42, n_init="auto")
                        val_pred = kmeans_val.fit_predict(z_val_np)
                        val_ari = adjusted_rand_score(val_labels, val_pred)
                
                self.refinement_head.train()
                self.dcam_memory.train()
                self.classifier.train()

                # ============================================================
                # Save ALL model parameters including criterion state
                # ============================================================
                if self.early_stop_metric == "ari":
                    if val_ari > best_val_ari:
                        best_val_ari = val_ari
                        patience_counter = 0
                        self.best_state = {
                            'refinement_head': self.refinement_head.state_dict().copy(),
                            'dcam_memory': self.dcam_memory.state_dict().copy(),
                            'classifier': self.classifier.state_dict().copy(),
                            'criterion': self.criterion.state_dict().copy(),
                        }
                    else:
                        patience_counter += 1
                else:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.best_state = {
                            'refinement_head': self.refinement_head.state_dict().copy(),
                            'dcam_memory': self.dcam_memory.state_dict().copy(),
                            'classifier': self.classifier.state_dict().copy(),
                            'criterion': self.criterion.state_dict().copy(),
                        }
                    else:
                        patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        best_val = best_val_ari if self.early_stop_metric == "ari" else best_val_loss
                        metric_name = "ARI" if self.early_stop_metric == "ari" else "loss"
                        print(f"Early stopping at epoch {epoch + 1} (best val {metric_name}: {best_val:.4f})")
                    # Restore best model
                    if self.best_state is not None:
                        self.refinement_head.load_state_dict(self.best_state['refinement_head'])
                        self.dcam_memory.load_state_dict(self.best_state['dcam_memory'])
                        self.classifier.load_state_dict(self.best_state['classifier'])
                        self.criterion.load_state_dict(self.best_state['criterion'])
                    break

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                ari_str = f"{val_ari:.4f}" if val_ari is not None else "N/A"
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Total: {epoch_losses['total']:.4f} | "
                      f"Triplet: {epoch_losses['triplet']:.4f} | "
                      f"CE: {epoch_losses['ce']:.4f} | "
                      f"DCAM: {epoch_losses['dcam']:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val ARI: {ari_str}")
                
                print(f"   Memory usage: active={memory_stats['total_active']}/{self.n_memories}, "
                      f"anchors={memory_stats['anchor_active']}/{self.n_anchor_slots}, "
                      f"free={memory_stats['free_active']}/{self.n_free_slots}")

        # Generate final embeddings
        self.refinement_head.eval()
        self.dcam_memory.eval()
        
        with torch.no_grad():
            final_embeddings = []
            for i in range(0, n_samples, batch_size):
                x_batch = embeddings_t[i:i+batch_size]
                z = self.refinement_head(x_batch)
                z_memory, _, _ = self.dcam_memory(z)
                final_embeddings.append(z_memory.cpu())
        
        return torch.cat(final_embeddings, dim=0).numpy()

    def transform(self, embeddings, batch_size=512):
        self.refinement_head.eval()
        self.dcam_memory.eval()
        embeddings_t = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        outputs = []
        with torch.no_grad():
            for i in range(0, embeddings_t.shape[0], batch_size):
                x_batch = embeddings_t[i:i+batch_size]
                z = self.refinement_head(x_batch)
                z_memory, _, _ = self.dcam_memory(z)
                outputs.append(z_memory.cpu())
        return torch.cat(outputs, dim=0).numpy()