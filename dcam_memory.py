"""
dcam_memory.py

DCAM-style Associative Memory module for scAMUSE (Production Ready)

Enhancements:
    - Idea 1: Labeled centroid initialization (pure torch, no numpy mixing)
    - Idea 3: Anchor vs free slot separation for subclusters

Features:
    - Iterative memory attractor updates
    - Attraction loss for pulling embeddings toward memories
    - Entropy, diversity, and utilization losses
    - Optional free-slot loss for subcluster capture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCAMemory(nn.Module):
    """
    DCAM-style Associative Memory with anchor/free slot architecture.
    
    Anchor slots: Initialized from labeled class centroids (Idea 1)
    Free slots: Capture subclusters/outliers (Idea 3)
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        n_memories: int = 64,
        n_anchor_slots: int = None,
        temperature: float = 0.5,
        update_strength: float = 0.5,
        n_iters: int = 3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_memories = n_memories
        self.temperature = temperature
        self.update_strength = update_strength
        self.n_iters = n_iters
        
        # Anchor vs free slot tracking (Idea 3)
        if n_anchor_slots is None:
            n_anchor_slots = n_memories
        self.n_anchor_slots = min(n_anchor_slots, n_memories)
        self.n_free_slots = n_memories - self.n_anchor_slots
        
        # Memory slots: anchor + free
        self.memories = nn.Parameter(torch.randn(n_memories, embedding_dim))
        nn.init.xavier_uniform_(self.memories)
        
        # Buffer to track which slots are anchors
        self.register_buffer(
            "is_anchor",
            torch.cat([
                torch.ones(self.n_anchor_slots, dtype=torch.bool),
                torch.zeros(self.n_free_slots, dtype=torch.bool)
            ])
        )

    def normalized_memories(self):
        """Return L2-normalized memory slots."""
        return F.normalize(self.memories, dim=1)
    
    def initialize_from_labels(self, embeddings, labels, idx_labeled, n_classes):
        """
        Idea 1: Initialize anchor slots from labeled class centroids.
        
        Args:
            embeddings: torch.Tensor (n_samples, embedding_dim) - ALREADY on correct device
            labels: torch.Tensor (n_samples,) - ALREADY on correct device  
            idx_labeled: torch.Tensor (n_labeled,) - indices of labeled samples, on correct device
            n_classes: int, number of classes
        
        NOTE: All inputs must be torch tensors on the SAME device as this module.
        """
        print(f"   Initializing {self.n_anchor_slots} anchor slots from labeled centroids...")
        
        device = self.memories.device
        
        # Validate inputs are tensors on correct device
        if not torch.is_tensor(embeddings):
            raise TypeError(f"embeddings must be torch.Tensor, got {type(embeddings)}")
        if not torch.is_tensor(labels):
            raise TypeError(f"labels must be torch.Tensor, got {type(labels)}")
        if not torch.is_tensor(idx_labeled):
            raise TypeError(f"idx_labeled must be torch.Tensor, got {type(idx_labeled)}")
        
        # Move to same device if needed
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        idx_labeled = idx_labeled.to(device)
        
        memory_init = self.memories.data.clone()
        valid_classes = 0
        
        # Extract labeled samples
        labeled_embeddings = embeddings[idx_labeled]
        labeled_labels = labels[idx_labeled]
        
        # Initialize anchor slots with class centroids
        n_to_init = min(self.n_anchor_slots, n_classes)
        for class_id in range(n_to_init):
            mask = (labeled_labels == class_id)
            n_samples_in_class = mask.sum().item()
            
            if n_samples_in_class > 0:
                # Pure torch: no numpy conversion
                centroid = labeled_embeddings[mask].mean(dim=0)
                memory_init[class_id] = F.normalize(centroid, dim=0)
                valid_classes += 1
        
        # Fill remaining anchor slots (if n_anchor_slots > n_classes)
        # Use first sample from cycling classes
        if self.n_anchor_slots > n_classes:
            for i in range(n_classes, self.n_anchor_slots):
                class_id = i % n_classes
                mask = (labeled_labels == class_id)
                
                if mask.sum() > 0:
                    # Get first matching sample
                    first_idx = torch.nonzero(mask, as_tuple=False)[0, 0]
                    memory_init[i] = F.normalize(labeled_embeddings[first_idx], dim=0)

        self.memories.data = memory_init
        
        print(f"   Initialized {valid_classes}/{n_to_init} anchor slots")
        
        if valid_classes == 0:
            print("   WARNING: No labeled data found for centroid initialization!")

    def forward(self, z: torch.Tensor):
        """
        Iterative attractor refinement of embeddings.
        
        Args:
            z: torch.Tensor (batch, embedding_dim) - assumed L2-normalized
        
        Returns:
            z_t: refined embeddings
            assignment: softmax assignment to memory slots
            energy: free energy (lower = more confident assignment)
        """
        # Input assumed normalized by RefinementHead. 
        # If not normalized, uncomment below:
        # z = F.normalize(z, dim=1)
        
        memories = self.normalized_memories()
        z_t = z
        
        # Iterative attractor dynamics
        for _ in range(self.n_iters):
            logits = torch.matmul(z_t, memories.T) / self.temperature
            assignment = F.softmax(logits, dim=1)
            memory_state = torch.matmul(assignment, memories)
            z_t = F.normalize(
                (1.0 - self.update_strength) * z_t + self.update_strength * memory_state,
                dim=1,
            )

        # Final assignment and energy
        logits = torch.matmul(z_t, memories.T) / self.temperature
        assignment = F.softmax(logits, dim=1)
        energy = -self.temperature * torch.logsumexp(logits, dim=1)

        return z_t, assignment, energy

    def energy_loss(self, energy: torch.Tensor):
        """
        Energy loss: minimize to make energy more negative (stronger attraction).
        Returns positive scalar to minimize.
        """
        return -energy.mean() / self.temperature

    def attraction_loss(self, z_memory: torch.Tensor, assignment: torch.Tensor):
        """
        Pull embeddings toward their assigned memory attractors.
        Uses cosine distance (1 - similarity).
        """
        memories = self.normalized_memories()
        target_states = torch.matmul(assignment, memories)
        distances = 1.0 - F.cosine_similarity(z_memory, target_states, dim=1)
        return distances.mean()

    def entropy_loss(self, assignment: torch.Tensor):
        """Encourage confident (low entropy) memory assignments."""
        entropy = -torch.sum(assignment * torch.log(assignment + 1e-8), dim=1)
        return entropy.mean()

    def diversity_loss(self):
        """
        Prevent memory collapse by spreading memory slots apart.
        Target: off-diagonal similarity = -1/(n-1) (orthogonal-like).
        """
        memories = self.normalized_memories()
        sim = torch.matmul(memories, memories.T)
        eye = torch.eye(self.n_memories, device=sim.device, dtype=sim.dtype)
        off_diag = sim * (1.0 - eye)
        target_sim = -1.0 / (self.n_memories - 1)
        return F.mse_loss(off_diag, torch.full_like(off_diag, target_sim))

    def utilization_loss(self, assignment: torch.Tensor):
        """Encourage uniform usage of ALL memory slots (anchors + free)."""
        usage = assignment.mean(dim=0)
        usage = torch.clamp(usage, min=1e-8)
        target = torch.ones_like(usage) / self.n_memories
        return F.kl_div(usage.log(), target, reduction='batchmean')
    
    def free_slot_loss(self, assignment: torch.Tensor):
        """
        Optional: Additional loss to encourage free slots to capture subclusters.
        Only penalizes free slot under-utilization.
        """
        if self.n_free_slots == 0:
            return torch.tensor(0.0, device=assignment.device)
        
        free_assignment = assignment[:, self.n_anchor_slots:]
        free_usage = free_assignment.mean(dim=0)
        free_usage = torch.clamp(free_usage, min=1e-8)
        target = torch.ones_like(free_usage) / self.n_free_slots
        
        return F.kl_div(free_usage.log(), target, reduction='batchmean')

    def dcam_loss(
        self,
        z_memory: torch.Tensor,
        energy: torch.Tensor,
        assignment: torch.Tensor,
        w_energy: float = 0.05,
        w_entropy: float = 0.2,
        w_diversity: float = 0.5,
        w_util: float = 0.1,
        w_attraction: float = 1.0,
        w_free_slots: float = 0.0,
    ):
        """
        Total DCAM memory loss.
        
        Args:
            z_memory: refined embeddings from forward()
            energy: free energy from forward()
            assignment: softmax assignment from forward()
            w_*: component weights (set to 0.0 to disable)
        """
        l_energy = self.energy_loss(energy)
        l_entropy = self.entropy_loss(assignment)
        l_diversity = self.diversity_loss()
        l_util = self.utilization_loss(assignment)
        l_attraction = self.attraction_loss(z_memory, assignment) if w_attraction > 0 else torch.tensor(0.0, device=z_memory.device)
        l_free_slots = self.free_slot_loss(assignment) if w_free_slots > 0 else torch.tensor(0.0, device=z_memory.device)

        total = (
            w_energy * l_energy
            + w_entropy * l_entropy
            + w_diversity * l_diversity
            + w_util * l_util
            + w_attraction * l_attraction
            + w_free_slots * l_free_slots
        )

        return {
            "dcam_total": total,
            "dcam_energy": l_energy,
            "dcam_entropy": l_entropy,
            "dcam_diversity": l_diversity,
            "dcam_utilization": l_util,
            "dcam_attraction": l_attraction,
            "dcam_free_slots": l_free_slots,
        }
    
    def get_memory_usage_stats(self, assignment: torch.Tensor):
        """
        Get statistics about memory slot usage.
        
        Args:
            assignment: torch.Tensor (n_samples, n_memories)
        """
        usage = assignment.mean(dim=0)
        anchor_usage = usage[:self.n_anchor_slots]
        free_usage = usage[self.n_anchor_slots:] if self.n_free_slots > 0 else torch.tensor([])
        
        stats = {
            "total_active": (usage > 0.01).sum().item(),
            "anchor_active": (anchor_usage > 0.01).sum().item(),
            "min_usage": usage.min().item(),
            "max_usage": usage.max().item(),
        }
        
        if self.n_free_slots > 0:
            stats["free_active"] = (free_usage > 0.01).sum().item()
            stats["free_usage_mean"] = free_usage.mean().item()
        else:
            stats["free_active"] = 0
            stats["free_usage_mean"] = 0.0
            
        return stats


__all__ = ["DCAMemory"]