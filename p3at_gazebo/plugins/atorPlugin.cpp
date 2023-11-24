#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
  class ActorPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Armazena o ponteiro para o modelo do ator
      this->model = _parent;

      // Define uma chamada de função para atualizar o estado do ator
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&ActorPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      // Obtém a posição atual do ator
      ignition::math::Pose3d pose = this->model->WorldPose();

      // Gera um vetor de deslocamento aleatório
      ignition::math::Vector3d randomDisplacement(
          ignition::math::Rand::DblUniform(-1, 1),
          ignition::math::Rand::DblUniform(-1, 1),
          0);

      // Define a velocidade linear do ator para o deslocamento aleatório
      double speed = 5.0; // Velocidade do ator (ajuste conforme necessário)
      ignition::math::Vector3d velocity = randomDisplacement.Normalized() * speed;
      this->model->SetLinearVel(velocity);

      // Atualiza a posição do ator
      this->model->SetWorldPose(pose);

      // Exibe a nova posição do ator
      std::cout << "Novo deslocamento: " << velocity << std::endl;
    }

    private: physics::ModelPtr model;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_MODEL_PLUGIN(ActorPlugin)
}

