#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cert_dir="${TYPEGPU_CERT_DIR:-$repo_root/certs}"
cert_host="${TYPEGPU_CERT_HOST:-localhost}"
cert_ip="${TYPEGPU_CERT_IP:-127.0.0.1}"

if [[ ! "$cert_host" =~ ^[A-Za-z0-9.-]+$ ]]; then
  printf 'Invalid certificate host: %s\n' "$cert_host" >&2
  exit 2
fi
if [[ ! "$cert_ip" =~ ^[0-9A-Fa-f:.]+$ ]]; then
  printf 'Invalid certificate IP: %s\n' "$cert_ip" >&2
  exit 2
fi

ca_key="$cert_dir/typegpu-preview-ca.key"
ca_cert="$cert_dir/typegpu-preview-ca.crt"
server_key="$cert_dir/preview-server.key"
server_csr="$cert_dir/preview-server.csr"
server_cert="$cert_dir/preview-server.crt"
serial_file="$cert_dir/typegpu-preview-ca.srl"
extensions="$cert_dir/preview-server.ext"

mkdir -p "$cert_dir"
chmod 700 "$cert_dir"
for target in "$ca_key" "$ca_cert" "$server_key" "$server_csr" "$server_cert"; do
  if [[ -e "$target" ]]; then
    printf 'Refusing to overwrite existing certificate material: %s\n' "$target" >&2
    exit 1
  fi
done

umask 077
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out "$ca_key"
openssl req -x509 -new -sha256 -days 90 -key "$ca_key" -out "$ca_cert" \
  -subj '/CN=TypeGPU Local Preview CA' \
  -addext 'basicConstraints=critical,CA:TRUE' \
  -addext 'keyUsage=critical,keyCertSign,cRLSign' \
  -addext 'subjectKeyIdentifier=hash'

printf '%s\n' \
  'authorityKeyIdentifier = keyid,issuer' \
  'basicConstraints = critical,CA:FALSE' \
  'keyUsage = critical,digitalSignature,keyEncipherment' \
  'extendedKeyUsage = serverAuth' \
  'subjectAltName = @alt_names' \
  '' \
  '[alt_names]' \
  "DNS.1 = $cert_host" \
  "IP.1 = $cert_ip" > "$extensions"

openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out "$server_key"
openssl req -new -sha256 -key "$server_key" -out "$server_csr" -subj "/CN=$cert_host"
openssl x509 -req -sha256 -days 90 -in "$server_csr" -CA "$ca_cert" -CAkey "$ca_key" \
  -CAcreateserial -CAserial "$serial_file" -out "$server_cert" -extfile "$extensions"

chmod 600 "$ca_key" "$server_key"
chmod 644 "$ca_cert" "$server_cert"
openssl verify -CAfile "$ca_cert" "$server_cert"
openssl x509 -in "$ca_cert" -noout -fingerprint -sha256 -dates
printf 'Trust only %s on devices used for this private preview.\n' "$ca_cert"
