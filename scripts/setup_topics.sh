#!/bin/bash
# Create all DriftSentinel Kafka topics.
# Run after docker-compose up when Kafka is healthy.
#
# Usage:
#   ./scripts/setup_topics.sh
#   # or:
#   make setup-topics

set -e

BOOTSTRAP=${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}

echo "Creating DriftSentinel Kafka topics on $BOOTSTRAP..."

create_topic() {
  local name=$1
  local partitions=$2
  local retention_ms=$3
  local extra=${4:-""}

  kafka-topics.sh \
    --bootstrap-server "$BOOTSTRAP" \
    --create --if-not-exists \
    --topic "$name" \
    --partitions "$partitions" \
    --replication-factor 1 \
    --config retention.ms="$retention_ms" \
    $extra \
    && echo "  ✓ $name"
}

create_topic "inference-events" 24 604800000 "--config compression.type=lz4"
create_topic "feature-stats"    12 2592000000 "--config cleanup.policy=compact --config compression.type=snappy"
create_topic "drift-alerts"     6  7776000000
create_topic "retrain-triggers" 6  604800000
create_topic "canary-decisions" 6  2592000000
create_topic "ge-quarantine"    6  1209600000
create_topic "incident-log"     3  31536000000

echo ""
echo "All topics created. Listing:"
kafka-topics.sh --bootstrap-server "$BOOTSTRAP" --list
