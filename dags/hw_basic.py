from datetime import timedelta

import pendulum
import requests
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sdk import BaseHook, Param, dag, task

import batch_common as core

MODEL_VERSION = "v1"
HTTP_TIMEOUT = 10


@dag(
    dag_id="hw_basic_batch_inference",
    start_date=pendulum.datetime(2026, 9, 1, tz="Europe/Moscow"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["homework", "basic"],
    params={
        "business_date": Param("2026-09-01", type="string", format="date"),
        "batch_size": Param(32, type="integer", minimum=1, maximum=1000),
    },
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=5),
        "execution_timeout": timedelta(minutes=2),
    },
)
def hw_basic_batch_inference():
    @task
    def extract(**context):
        day = core.business_date(context["params"]["business_date"], context.get("data_interval_start"))
        host = BaseHook.get_connection("source_api").host

        data_response = requests.get(f"{host}/basic/{day}.json", timeout=HTTP_TIMEOUT)
        data_response.raise_for_status()
        data = data_response.json()
        if data.get("business_date") != day:
            raise ValueError(f"Source returned {data.get('business_date')!r} instead of {day}")

        model_response = requests.get(f"{host}/models/{MODEL_VERSION}.json", timeout=HTTP_TIMEOUT)
        model_response.raise_for_status()

        return core.save({
            "business_date": day,
            "rows": data["rows"],
            "model": model_response.json(),
            "model_sha256": core.digest(model_response.content),
        })

    @task(retries=0)
    def infer(ref, **context):
        artifact = core.read(ref)
        day = artifact["business_date"]
        rows = core.validate_rows(artifact["rows"], day)
        predictions = core.infer_batches(rows, artifact["model"], context["params"]["batch_size"])
        core.validate_predictions(predictions, [row["customer_id"] for row in rows])
        return core.save({
            "business_date": day,
            "model_version": MODEL_VERSION,
            "predictions": predictions,
            "model_sha256": artifact["model_sha256"],
        })

    @task
    def publish(ref):
        result = core.read(ref)
        conn = PostgresHook(postgres_conn_id="warehouse").get_conn()
        try:
            core.replace_day(conn, result["business_date"], result["model_version"], result["predictions"])
        finally:
            conn.close()
        return {
            "business_date": result["business_date"],
            "model_version": result["model_version"],
            "rows": len(result["predictions"]),
        }

    publish(infer(extract()))


hw_basic_batch_inference()
