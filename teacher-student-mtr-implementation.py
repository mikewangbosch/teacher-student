import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional

class TeacherStudentMTR(nn.Module):
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        noise_levels: List[float] = [0.1, 0.3, 0.5, 0.7],
        temperature: float = 1.0,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.noise_levels = noise_levels
        self.temperature = temperature
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Additional heads for student model
        self.denoising_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)  # x, y coordinates
            ) for _ in noise_levels
        ])
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # scalar uncertainty
        )
        
    def add_noise(self, trajectories: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add Gaussian noise to trajectories."""
        noise = torch.randn_like(trajectories) * noise_level
        return trajectories + noise
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through both teacher and student."""
        with torch.no_grad():
            teacher_out = self.teacher(batch)
            
        student_out = self.student(batch)
        
        # Generate noisy trajectories for denoising
        noisy_trajectories = {
            level: self.add_noise(teacher_out['trajectories'], level)
            for level in self.noise_levels
        }
        
        # Predict denoised trajectories
        denoised_predictions = {
            level: self.denoising_head[i](self.student.get_features(batch))
            for i, level in enumerate(self.noise_levels)
        }
        
        # Predict uncertainty
        uncertainty = self.uncertainty_head(self.student.get_features(batch))
        
        return {
            'teacher': teacher_out,
            'student': student_out,
            'noisy': noisy_trajectories,
            'denoised': denoised_predictions,
            'uncertainty': uncertainty
        }
    
    def distribution_matching_loss(
        self,
        teacher_trajectories: torch.Tensor,
        student_trajectories: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between teacher and student distributions."""
        # Convert trajectories to distributions
        teacher_dist = F.softmax(teacher_trajectories / self.temperature, dim=-1)
        student_dist = F.log_softmax(student_trajectories / self.temperature, dim=-1)
        
        return F.kl_div(student_dist, teacher_dist, reduction='batchmean')
    
    def denoising_matching_loss(
        self,
        teacher_trajectories: torch.Tensor,
        denoised_predictions: Dict[float, torch.Tensor]
    ) -> torch.Tensor:
        """Compute MSE between teacher's clean and student's denoised trajectories."""
        loss = 0.0
        for level, pred in denoised_predictions.items():
            loss += F.mse_loss(pred, teacher_trajectories)
        return loss / len(self.noise_levels)
    
    def joint_prediction_loss(
        self,
        teacher_trajectories: torch.Tensor,
        student_trajectories: torch.Tensor,
        agent_pairs: torch.Tensor
    ) -> torch.Tensor:
        """Compute consistency loss for joint predictions."""
        teacher_joints = self.extract_joint_predictions(teacher_trajectories, agent_pairs)
        student_joints = self.extract_joint_predictions(student_trajectories, agent_pairs)
        
        return F.mse_loss(student_joints, teacher_joints)
    
    def uncertainty_alignment_loss(
        self,
        teacher_uncertainty: torch.Tensor,
        student_uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """Align uncertainty estimates between teacher and student."""
        return F.mse_loss(student_uncertainty, teacher_uncertainty)
    
    def extract_joint_predictions(
        self,
        trajectories: torch.Tensor,
        agent_pairs: torch.Tensor
    ) -> torch.Tensor:
        """Extract pairwise joint predictions between agents."""
        joints = []
        for pair in agent_pairs:
            agent1, agent2 = pair
            joint = torch.cat([
                trajectories[agent1],
                trajectories[agent2]
            ], dim=-1)
            joints.append(joint)
        return torch.stack(joints)

class TeacherStudentTrainer:
    def __init__(
        self,
        model: TeacherStudentMTR,
        optimizer: torch.optim.Optimizer,
        loss_weights: Dict[str, float]
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_weights = loss_weights
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute single training step."""
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch)
        
        # Compute losses
        losses = {}
        
        # Distribution matching loss
        losses['dist'] = self.model.distribution_matching_loss(
            outputs['teacher']['trajectories'],
            outputs['student']['trajectories']
        )
        
        # Denoising matching loss
        losses['denoise'] = self.model.denoising_matching_loss(
            outputs['teacher']['trajectories'],
            outputs['denoised']
        )
        
        # Joint prediction loss
        losses['joint'] = self.model.joint_prediction_loss(
            outputs['teacher']['trajectories'],
            outputs['student']['trajectories'],
            batch['agent_pairs']
        )
        
        # Uncertainty alignment loss
        losses['uncertainty'] = self.model.uncertainty_alignment_loss(
            outputs['teacher']['uncertainty'],
            outputs['uncertainty']
        )
        
        # Compute weighted total loss
        total_loss = sum(
            self.loss_weights[k] * v for k, v in losses.items()
        )
        
        # Backward pass and optimize
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss,
            **losses
        }

# Example usage
if __name__ == "__main__":
    # Initialize models
    teacher = DiffusionModel()  # Your diffusion model
    student = MTRModel()        # Your MTR model
    
    # Create teacher-student framework
    model = TeacherStudentMTR(
        teacher_model=teacher,
        student_model=student,
        noise_levels=[0.1, 0.3, 0.5, 0.7],
        temperature=1.0,
        hidden_dim=256
    )
    
    # Setup trainer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )
    
    trainer = TeacherStudentTrainer(
        model=model,
        optimizer=optimizer,
        loss_weights={
            'dist': 1.0,
            'denoise': 0.5,
            'joint': 1.0,
            'uncertainty': 0.5
        }
    )
    
    # Training loop
    for batch in dataloader:
        losses = trainer.train_step(batch)