"""
Elara — Baseline coherent sessions (n=10)
Runs 10 deepening conversations without topic breaks.
Proxy must be running: python elara/proxy.py
Set ANTHROPIC_BASE_URL=http://localhost:8877 before running.
"""

import anthropic
import time

client = anthropic.Anthropic()

SESSIONS = [
    {
        "id": 21, "topic": "algebra_lineal",
        "turns": [
            "¿Qué es un espacio vectorial y por qué es útil?",
            "¿Cómo se relacionan los vectores con las transformaciones lineales?",
            "¿Qué significa que una transformación sea invertible?",
            "¿Cómo conecta esto con los valores y vectores propios?",
            "¿Para qué sirven los valores propios en aplicaciones reales?",
        ]
    },
    {
        "id": 22, "topic": "evolucion_biologica",
        "turns": [
            "¿Cómo funciona la selección natural de Darwin?",
            "¿Qué papel juegan las mutaciones en la evolución?",
            "¿Cómo se forma una nueva especie a partir de otra?",
            "¿Qué evidencia tenemos de la evolución humana?",
            "¿Cómo explica la evolución la complejidad del ojo?",
        ]
    },
    {
        "id": 23, "topic": "termodinamica",
        "turns": [
            "¿Qué dice la primera ley de la termodinámica?",
            "¿Por qué la segunda ley implica que el desorden siempre aumenta?",
            "¿Qué es la entropía en términos prácticos?",
            "¿Cómo funcionan los motores de calor a partir de estas leyes?",
            "¿Cuál es el límite teórico de eficiencia de un motor?",
        ]
    },
    {
        "id": 24, "topic": "linguistica_historica",
        "turns": [
            "¿Cómo cambian los idiomas con el tiempo?",
            "¿Qué es el proto-indoeuropeo y cómo lo reconstruimos?",
            "¿Cómo se forman familias de lenguas?",
            "¿Qué tienen en común el español, el griego y el sánscrito?",
            "¿Cómo sabemos cómo sonaba el latín clásico?",
        ]
    },
    {
        "id": 25, "topic": "teoria_de_grafos",
        "turns": [
            "¿Qué es un grafo y para qué sirve?",
            "¿Cuál es la diferencia entre un grafo dirigido y uno no dirigido?",
            "¿Qué es un árbol en teoría de grafos?",
            "¿Cómo funciona el algoritmo de Dijkstra?",
            "¿Para qué se usan los grafos en redes sociales?",
        ]
    },
    {
        "id": 26, "topic": "psicologia_cognitiva",
        "turns": [
            "¿Cómo procesa el cerebro la información visual?",
            "¿Qué diferencia hay entre memoria de trabajo y memoria de largo plazo?",
            "¿Qué es el efecto de primacía y recencia en la memoria?",
            "¿Cómo afectan los sesgos cognitivos a la toma de decisiones?",
            "¿Qué es el sesgo de confirmación y cómo se evita?",
        ]
    },
    {
        "id": 27, "topic": "relatividad_especial",
        "turns": [
            "¿Por qué la velocidad de la luz es el límite universal?",
            "¿Qué significa que el tiempo sea relativo?",
            "¿Cómo funciona la dilatación temporal?",
            "¿Qué es la paradoja de los gemelos?",
            "¿Cómo se relaciona E=mc² con la relatividad especial?",
        ]
    },
    {
        "id": 28, "topic": "economia_conductual",
        "turns": [
            "¿En qué se diferencia la economía conductual de la clásica?",
            "¿Qué son los heurísticos y por qué nos llevan a errores?",
            "¿Qué demostró Kahneman con la teoría de la perspectiva?",
            "¿Cómo funciona el nudge en políticas públicas?",
            "¿Qué experimentos famosos demuestran la irracionalidad humana?",
        ]
    },
    {
        "id": 29, "topic": "logica_formal",
        "turns": [
            "¿Qué es una proposición lógica y cómo se evalúa?",
            "¿Cuál es la diferencia entre deducción e inducción?",
            "¿Qué es un silogismo y cuándo es válido?",
            "¿Cómo funciona la lógica de predicados?",
            "¿Qué dice el teorema de incompletitud de Gödel?",
        ]
    },
    {
        "id": 30, "topic": "mecanica_clasica",
        "turns": [
            "¿Qué dicen las tres leyes de Newton?",
            "¿Cómo se relacionan fuerza, masa y aceleración en la práctica?",
            "¿Qué es la energía cinética y cómo se conserva?",
            "¿Qué es el momento angular y por qué se conserva?",
            "¿Cómo describe el hamiltoniano un sistema mecánico?",
        ]
    },
]

def run_session(session):
    sid = session["id"]
    topic = session["topic"]
    print(f"\n{'='*50}")
    print(f"Session {sid} — {topic}")
    print(f"{'='*50}")

    history = []
    for i, question in enumerate(session["turns"]):
        print(f"  Turn {i+1}: {question[:60]}...")
        history.append({"role": "user", "content": question})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=history,
        )
        answer = response.content[0].text
        history.append({"role": "assistant", "content": answer})
        print(f"  → {answer[:80]}...")
        time.sleep(1)  # avoid rate limits

    print(f"  Session {sid} complete — {len(session['turns'])} turns")

if __name__ == "__main__":
    print("Elara baseline coherent sessions")
    print("Proxy must be running at localhost:8877")
    print(f"Running {len(SESSIONS)} sessions...\n")

    for session in SESSIONS:
        run_session(session)
        time.sleep(2)

    print("\nAll baseline sessions complete.")
    print("Check elara_sessions.db for recorded metrics.")
