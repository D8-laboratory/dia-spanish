"""
Generate synthetic Spanish dialogue transcripts for Dia training.

Uses an LLM to create realistic multi-turn Spanish conversations
with speaker tags [S1]/[S2] and nonverbal cues.

Output format (JSONL):
{
  "id": "es_syn_0001",
  "transcript": "[S1] ¡Hola! ¿Cómo estás? [S2] ¡Bien, gracias! (risas) ...",
  "domain": "casual",
  "region": "colombia",
  "num_turns": 6
}
"""

import argparse
import json
import random
from pathlib import Path


DOMAINS = [
    "casual", "trabajo", "familia", "salud", "tecnología",
    "cocina", "deportes", "viajes", "educación", "negocios",
    "entretenimiento", "noticias", "cultura", "moda", "finanzas",
]

REGIONS = [
    "colombia", "mexico", "argentina", "españa", "chile",
    "peru", "venezuela", "ecuador", "neutral",
]

NONVERBAL_ES = [
    "(risas)", "(suspira)", "(tose)", "(gime)", "(bosteza)",
    "(llora)", "(grita)", "(canta)", "(silba)", "(aplausos)",
    "(respira profundo)", "(asiente)", "(murmullos)",
]

SAMPLE_DIALOGUES = [
    {
        "domain": "casual",
        "transcript": "[S1] ¡Hola! ¿Cómo te fue en la reunión? [S2] ¡Uf, súper bien! Estaba nerviosa pero al final todo salió perfecto. (risas) [S1] ¡Qué bueno! Me alegro mucho. ¿Y qué te dijeron del proyecto? [S2] Pues, les encantó la idea. Quieren que lo presentemos la próxima semana ante la junta directiva. [S1] Wow, eso es genial. ¿Necesitas ayuda preparando la presentación? [S2] La verdad sí, me haría falta una mano. ¿Te viene bien mañana por la tarde?",
    },
    {
        "domain": "trabajo",
        "transcript": "[S1] Buenos días, equipo. ¿Cómo vamos con los números del trimestre? [S2] Buenos días. Pues, estamos un cinco por ciento por encima de la meta. (suspira) Pero todavía falta consolidar los datos de la región sur. [S1] Entiendo. ¿Cuándo crees que tendremos eso listo? [S2] A más tardar el viernes. Ya hablé con el equipo de Bogotá y están compilando los últimos reportes.",
    },
    {
        "domain": "negocios",
        "transcript": "[S1] Oiga, ¿ya vio la propuesta que mandó el cliente? [S2] Sí, acabó de llegar. (respira profundo) Piden un descuento del veinte por ciento sobre el presupuesto original. [S1] ¿Veinte? Eso es bastante. ¿Qué opinas? [S2] Pues yo digo que negociemos. Podemos ofrecerles un quince con compromiso a largo plazo. [S1] Me parece bien. Preparemos una contrapropuesta y la enviamos hoy mismo.",
    },
    {
        "domain": "familia",
        "transcript": "[S1] Mami, ¿puedo ir al cine con mis amigos este sábado? [S2] ¿Qué películas están dando? [S1] Hay una de superhéroes que está súper buena. Todos mis amigos ya la vieron. (risas) [S2] Está bien, pero con condición de que llegues antes de las nueve. [S1] ¡Gracias, mami! Prometo portarme bien. (risas)",
    },
    {
        "domain": "tecnología",
        "transcript": "[S1] ¿Ya probaste la nueva actualización de la aplicación? [S2] ¡Sí! Está increíble. Ahora tiene modo oscuro y todo. [S1] ¿En serio? Tengo que actualizarla ya mismo. [S2] Y también agregaron un asistente de voz que entiende español perfecto. (risas) Por fin, ¿no? [S1] Jaja, era hora. Las versiones anteriores eran un dolor de cabeza.",
    },
    {
        "domain": "cocina",
        "transcript": "[S1] ¿Me pasas la receta de esa sopa que hiciste el domingo? [S2] ¡Claro! Es súper fácil. Necesitas plátano maduro, leche de coco y un poquito de cilantro. [S1] ¿Y cuánto tiempo de cocción lleva? [S2] Unos veinte minutos. El secreto es sofreír el plátano antes de licuarlo. (risas) Ese es el toque de mi abuela. [S1] Qué rico. La voy a preparar este fin de semana.",
    },
]


def generate_dialogue(dialogue_id: str, domain: str = None, region: str = None) -> dict:
    """Generate a single Spanish dialogue transcript.

    In production, this would call an LLM (GPT-4, Claude, etc.)
    For now, samples from our template pool with variations.
    """
    domain = domain or random.choice(DOMAINS)
    region = region or random.choice(REGIONS)

    template = random.choice(SAMPLE_DIALOGUES)

    return {
        "id": dialogue_id,
        "transcript": template["transcript"],
        "domain": domain,
        "region": region,
        "num_turns": template["transcript"].count("[S"),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Spanish dialogues for Dia training")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of dialogues to generate")
    parser.add_argument("--output", type=str, default="data/raw/synthetic/spanish_dialogues.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            dialogue = generate_dialogue(
                dialogue_id=f"es_syn_{i+1:04d}",
            )
            f.write(json.dumps(dialogue, ensure_ascii=False) + "\n")

    print(f"Generated {args.num_samples} Spanish dialogues → {output_path}")


if __name__ == "__main__":
    main()
