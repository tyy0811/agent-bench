A detailed look at the different policy levels defined in the Pod Security Standards.

The Pod Security Standards define three different *policies* to broadly cover the security spectrum. These policies are *cumulative* and range from highly-permissive to highly-restrictive. This guide outlines the requirements of each policy.

| Profile | Description |
| --- | --- |
| **Privileged** | Unrestricted policy, providing the widest possible level of permissions. This policy allows for known privilege escalations. |
| **Baseline** | Minimally restrictive policy which prevents known privilege escalations. Allows the default (minimally specified) Pod configuration. |
| **Restricted** | Heavily restricted policy, following current Pod hardening best practices. |

## Profile Details

### Privileged

**The *Privileged* policy is purposely-open, and entirely unrestricted.** This type of policy is typically aimed at system- and infrastructure-level workloads managed by privileged, trusted users.

The Privileged policy is defined by an absence of restrictions. If you define a Pod where the Privileged security policy applies, the Pod you define is able to bypass typical container isolation mechanisms. For example, you can define a Pod that has access to the node's host network.

### Baseline

**The *Baseline* policy is aimed at ease of adoption for common containerized workloads while preventing known privilege escalations.** This policy is targeted at application operators and developers of non-critical applications. The following listed controls should be enforced/disallowed:

> [!info] Note:
> In this table, wildcards (`*`) indicate all elements in a list. For example, `spec.containers[*].securityContext` refers to the Security Context object for *all defined containers*. If any of the listed containers fails to meet the requirements, the entire pod will fail validation.

| Control | Policy |
| --- | --- |
| HostProcess | Windows Pods offer the ability to run [HostProcess containers](https://kubernetes.io/docs/tasks/configure-pod-container/create-hostprocess-pod) which enables privileged access to the Windows host machine. Privileged access to the host is disallowed in the Baseline policy.  FEATURE STATE: `Kubernetes v1.26 [stable]`  **Restricted Fields**  - `spec.securityContext.windowsOptions.hostProcess` - `spec.containers[*].securityContext.windowsOptions.hostProcess` - `spec.initContainers[*].securityContext.windowsOptions.hostProcess` - `spec.ephemeralContainers[*].securityContext.windowsOptions.hostProcess`  **Allowed Values**  - Undefined/nil - `false` |
| Host Namespaces | Sharing the host namespaces must be disallowed.  **Restricted Fields**  - `spec.hostNetwork` - `spec.hostPID` - `spec.hostIPC`  **Allowed Values**  - Undefined/nil - `false` |
| Privileged Containers | Privileged Pods disable most security mechanisms and must be disallowed.  **Restricted Fields**  - `spec.containers[*].securityContext.privileged` - `spec.initContainers[*].securityContext.privileged` - `spec.ephemeralContainers[*].securityContext.privileged`  **Allowed Values**  - Undefined/nil - `false` |
| Capabilities | Adding additional capabilities beyond those listed below must be disallowed.  **Restricted Fields**  - `spec.containers[*].securityContext.capabilities.add` - `spec.initContainers[*].securityContext.capabilities.add` - `spec.ephemeralContainers[*].securityContext.capabilities.add`  **Allowed Values**  - Undefined/nil - `AUDIT_WRITE` - `CHOWN` - `DAC_OVERRIDE` - `FOWNER` - `FSETID` - `KILL` - `MKNOD` - `NET_BIND_SERVICE` - `SETFCAP` - `SETGID` - `SETPCAP` - `SETUID` - `SYS_CHROOT` |
| HostPath Volumes | HostPath volumes must be forbidden.  **Restricted Fields**  - `spec.volumes[*].hostPath`  **Allowed Values**  - Undefined/nil |
| Host Ports | HostPorts should be disallowed entirely (recommended) or restricted to a known list  **Restricted Fields**  - `spec.containers[*].ports[*].hostPort` - `spec.initContainers[*].ports[*].hostPort` - `spec.ephemeralContainers[*].ports[*].hostPort`  **Allowed Values**  - Undefined/nil - Known list (not supported by the built-in [Pod Security Admission controller](https://kubernetes.io/docs/concepts/security/pod-security-admission/)) - `0` |
| Host Probes / Lifecycle Hooks (v1.34+) | The Host field in probes and lifecycle hooks must be disallowed.  **Restricted Fields**  - `spec.containers[*].livenessProbe.httpGet.host` - `spec.containers[*].readinessProbe.httpGet.host` - `spec.containers[*].startupProbe.httpGet.host` - `spec.containers[*].livenessProbe.tcpSocket.host` - `spec.containers[*].readinessProbe.tcpSocket.host` - `spec.containers[*].startupProbe.tcpSocket.host` - `spec.containers[*].lifecycle.postStart.tcpSocket.host` - `spec.containers[*].lifecycle.preStop.tcpSocket.host` - `spec.containers[*].lifecycle.postStart.httpGet.host` - `spec.containers[*].lifecycle.preStop.httpGet.host` - `spec.initContainers[*].livenessProbe.httpGet.host` - `spec.initContainers[*].readinessProbe.httpGet.host` - `spec.initContainers[*].startupProbe.httpGet.host` - `spec.initContainers[*].livenessProbe.tcpSocket.host` - `spec.initContainers[*].readinessProbe.tcpSocket.host` - `spec.initContainers[*].startupProbe.tcpSocket.host` - `spec.initContainers[*].lifecycle.postStart.tcpSocket.host` - `spec.initContainers[*].lifecycle.preStop.tcpSocket.host` - `spec.initContainers[*].lifecycle.postStart.httpGet.host` - `spec.initContainers[*].lifecycle.preStop.httpGet.host`  **Allowed Values**  - Undefined/nil - "" |
| AppArmor | On supported hosts, the `RuntimeDefault` AppArmor profile is applied by default. The baseline policy should prevent overriding or disabling the default AppArmor profile, or restrict overrides to an allowed set of profiles.  **Restricted Fields**  - `spec.securityContext.appArmorProfile.type` - `spec.containers[*].securityContext.appArmorProfile.type` - `spec.initContainers[*].securityContext.appArmorProfile.type` - `spec.ephemeralContainers[*].securityContext.appArmorProfile.type`  **Allowed Values**  - Undefined/nil - `RuntimeDefault` - `Localhost`  ---  - `metadata.annotations["container.apparmor.security.beta.kubernetes.io/*"]`  **Allowed Values**  - Undefined/nil - `runtime/default` - `localhost/*` |
| SELinux | Setting the SELinux type is restricted, and setting a custom SELinux user or role option is forbidden.  **Restricted Fields**  - `spec.securityContext.seLinuxOptions.type` - `spec.containers[*].securityContext.seLinuxOptions.type` - `spec.initContainers[*].securityContext.seLinuxOptions.type` - `spec.ephemeralContainers[*].securityContext.seLinuxOptions.type`  **Allowed Values**  - Undefined/"" - `container_t` - `container_init_t` - `container_kvm_t` - `container_engine_t` (since Kubernetes 1.31)  ---  **Restricted Fields**  - `spec.securityContext.seLinuxOptions.user` - `spec.containers[*].securityContext.seLinuxOptions.user` - `spec.initContainers[*].securityContext.seLinuxOptions.user` - `spec.ephemeralContainers[*].securityContext.seLinuxOptions.user` - `spec.securityContext.seLinuxOptions.role` - `spec.containers[*].securityContext.seLinuxOptions.role` - `spec.initContainers[*].securityContext.seLinuxOptions.role` - `spec.ephemeralContainers[*].securityContext.seLinuxOptions.role`  **Allowed Values**  - Undefined/"" |
| `/proc` Mount Type | The default `/proc` masks are set up to reduce attack surface, and should be required.  **Restricted Fields**  - `spec.containers[*].securityContext.procMount` - `spec.initContainers[*].securityContext.procMount` - `spec.ephemeralContainers[*].securityContext.procMount`  **Allowed Values**  - Undefined/nil - `Default` |
| Seccomp | Seccomp profile must not be explicitly set to `Unconfined`.  **Restricted Fields**  - `spec.securityContext.seccompProfile.type` - `spec.containers[*].securityContext.seccompProfile.type` - `spec.initContainers[*].securityContext.seccompProfile.type` - `spec.ephemeralContainers[*].securityContext.seccompProfile.type`  **Allowed Values**  - Undefined/nil - `RuntimeDefault` - `Localhost` |
| Sysctls | Sysctls can disable security mechanisms or affect all containers on a host, and should be disallowed except for an allowed "safe" subset. A sysctl is considered safe if it is namespaced in the container or the Pod, and it is isolated from other Pods or processes on the same Node.  **Restricted Fields**  - `spec.securityContext.sysctls[*].name`  **Allowed Values**  - Undefined/nil - `kernel.shm_rmid_forced` - `net.ipv4.ip_local_port_range` - `net.ipv4.ip_unprivileged_port_start` - `net.ipv4.tcp_syncookies` - `net.ipv4.ping_group_range` - `net.ipv4.ip_local_reserved_ports` (since Kubernetes 1.27) - `net.ipv4.tcp_keepalive_time` (since Kubernetes 1.29) - `net.ipv4.tcp_fin_timeout` (since Kubernetes 1.29) - `net.ipv4.tcp_keepalive_intvl` (since Kubernetes 1.29) - `net.ipv4.tcp_keepalive_probes` (since Kubernetes 1.29) |

### Restricted

**The *Restricted* policy is aimed at enforcing current Pod hardening best practices, at the expense of some compatibility.** It is targeted at operators and developers of security-critical applications, as well as lower-trust users. The following listed controls should be enforced/disallowed:

> [!info] Note:
> In this table, wildcards (`*`) indicate all elements in a list. For example, `spec.containers[*].securityContext` refers to the Security Context object for *all defined containers*. If any of the listed containers fails to meet the requirements, the entire pod will fail validation.

<table><tbody><tr><td><strong>Control</strong></td><td><strong>Policy</strong></td></tr><tr><td colspan="2"><em>Everything from the Baseline policy</em></td></tr><tr><td>Volume Types</td><td><p>The Restricted policy only permits the following volume types.</p><p><strong>Restricted Fields</strong></p><ul><li><code>spec.volumes[*]</code></li></ul><p><strong>Allowed Values</strong></p>Every item in the <code>spec.volumes[*]</code> list must set one of the following fields to a non-null value:<ul><li><code>spec.volumes[*].configMap</code></li><li><code>spec.volumes[*].csi</code></li><li><code>spec.volumes[*].downwardAPI</code></li><li><code>spec.volumes[*].emptyDir</code></li><li><code>spec.volumes[*].ephemeral</code></li><li><code>spec.volumes[*].persistentVolumeClaim</code></li><li><code>spec.volumes[*].projected</code></li><li><code>spec.volumes[*].secret</code></li></ul></td></tr><tr><td>Privilege Escalation (v1.8+)</td><td><p>Privilege escalation (such as via set-user-ID or set-group-ID file mode) should not be allowed. <em><a href="#os-specific-policy-controls">This is Linux only policy</a> in v1.25+ <code>(spec.os.name != windows)</code></em></p><p><strong>Restricted Fields</strong></p><ul><li><code>spec.containers[*].securityContext.allowPrivilegeEscalation</code></li><li><code>spec.initContainers[*].securityContext.allowPrivilegeEscalation</code></li><li><code>spec.ephemeralContainers[*].securityContext.allowPrivilegeEscalation</code></li></ul><p><strong>Allowed Values</strong></p><ul><li><code>false</code></li></ul></td></tr><tr><td>Running as Non-root</td><td><p>Containers must be required to run as non-root users.</p><p><strong>Restricted Fields</strong></p><ul><li><code>spec.securityContext.runAsNonRoot</code></li><li><code>spec.containers[*].securityContext.runAsNonRoot</code></li><li><code>spec.initContainers[*].securityContext.runAsNonRoot</code></li><li><code>spec.ephemeralContainers[*].securityContext.runAsNonRoot</code></li></ul><p><strong>Allowed Values</strong></p><ul><li><code>true</code></li></ul><small>The container fields may be undefined/ <code>nil</code> if the pod-level <code>spec.securityContext.runAsNonRoot</code> is set to <code>true</code>.</small></td></tr><tr><td>Running as Non-root user (v1.23+)</td><td><p>Containers must not set <tt>runAsUser</tt> to 0</p><p><strong>Restricted Fields</strong></p><ul><li><code>spec.securityContext.runAsUser</code></li><li><code>spec.containers[*].securityContext.runAsUser</code></li><li><code>spec.initContainers[*].securityContext.runAsUser</code></li><li><code>spec.ephemeralContainers[*].securityContext.runAsUser</code></li></ul><p><strong>Allowed Values</strong></p><ul><li>any non-zero value</li><li><code>undefined/null</code></li></ul></td></tr><tr><td>Seccomp (v1.19+)</td><td><p>Seccomp profile must be explicitly set to one of the allowed values. Both the <code>Unconfined</code> profile and the <em>absence</em> of a profile are prohibited. <em><a href="#os-specific-policy-controls">This is Linux only policy</a> in v1.25+ <code>(spec.os.name != windows)</code></em></p><p><strong>Restricted Fields</strong></p><ul><li><code>spec.securityContext.seccompProfile.type</code></li><li><code>spec.containers[*].securityContext.seccompProfile.type</code></li><li><code>spec.initContainers[*].securityContext.seccompProfile.type</code></li><li><code>spec.ephemeralContainers[*].securityContext.seccompProfile.type</code></li></ul><p><strong>Allowed Values</strong></p><ul><li><code>RuntimeDefault</code></li><li><code>Localhost</code></li></ul><small>The container fields may be undefined/ <code>nil</code> if the pod-level <code>spec.securityContext.seccompProfile.type</code> field is set appropriately. Conversely, the pod-level field may be undefined/ <code>nil</code> if _all_ container- level fields are set.</small></td></tr><tr><td>Capabilities (v1.22+)</td><td><p>Containers must drop <code>ALL</code> capabilities, and are only permitted to add back the <code>NET_BIND_SERVICE</code> capability. <em><a href="#os-specific-policy-controls">This is Linux only policy</a> in v1.25+ <code>(.spec.os.name != "windows")</code></em></p><p><strong>Restricted Fields</strong></p><ul><li><code>spec.containers[*].securityContext.capabilities.drop</code></li><li><code>spec.initContainers[*].securityContext.capabilities.drop</code></li><li><code>spec.ephemeralContainers[*].securityContext.capabilities.drop</code></li></ul><p><strong>Allowed Values</strong></p><ul><li>Any list of capabilities that includes <code>ALL</code></li></ul><hr><p><strong>Restricted Fields</strong></p><ul><li><code>spec.containers[*].securityContext.capabilities.add</code></li><li><code>spec.initContainers[*].securityContext.capabilities.add</code></li><li><code>spec.ephemeralContainers[*].securityContext.capabilities.add</code></li></ul><p><strong>Allowed Values</strong></p><ul><li>Undefined/nil</li><li><code>NET_BIND_SERVICE</code></li></ul></td></tr></tbody></table>

## Policy Instantiation

Decoupling policy definition from policy instantiation allows for a common understanding and consistent language of policies across clusters, independent of the underlying enforcement mechanism.

As mechanisms mature, they will be defined below on a per-policy basis. The methods of enforcement of individual policies are not defined here.

[**Pod Security Admission Controller**](https://kubernetes.io/docs/concepts/security/pod-security-admission/)

- [Privileged namespace](https://raw.githubusercontent.com/kubernetes/website/main/content/en/examples/security/podsecurity-privileged.yaml)
- [Baseline namespace](https://raw.githubusercontent.com/kubernetes/website/main/content/en/examples/security/podsecurity-baseline.yaml)
- [Restricted namespace](https://raw.githubusercontent.com/kubernetes/website/main/content/en/examples/security/podsecurity-restricted.yaml)

### Alternatives

> [!secondary] Secondary
> **Note:** This section links to third party projects that provide functionality required by Kubernetes. The Kubernetes project authors aren't responsible for these projects, which are listed alphabetically. To add a project to this list, read the [content guide](https://kubernetes.io/docs/contribute/style/content-guide/#third-party-content) before submitting a change. [More information.](#third-party-content-disclaimer)

Other alternatives for enforcing policies are being developed in the Kubernetes ecosystem, such as:

- [Kubewarden](https://github.com/kubewarden)
- [Kyverno](https://kyverno.io/policies/pod-security/)
- [OPA Gatekeeper](https://github.com/open-policy-agent/gatekeeper)

## Pod OS field

Kubernetes lets you use nodes that run either Linux or Windows. You can mix both kinds of node in one cluster. Windows in Kubernetes has some limitations and differentiators from Linux-based workloads. Specifically, many of the Pod `securityContext` fields [have no effect on Windows](https://kubernetes.io/docs/concepts/windows/intro/#compatibility-v1-pod-spec-containers-securitycontext).

> [!info] Note:
> Kubelets prior to v1.24 don't enforce the pod OS field, and if a cluster has nodes on versions earlier than v1.24 the Restricted policies should be pinned to a version prior to v1.25.

### Restricted Pod Security Standard changes

Another important change, made in Kubernetes v1.25 is that the *Restricted* policy has been updated to use the `pod.spec.os.name` field. Based on the OS name, certain policies that are specific to a particular OS can be relaxed for the other OS.

#### OS-specific policy controls

Restrictions on the following controls are only required if `.spec.os.name` is not `windows`:

- Privilege Escalation
- Seccomp
- Linux Capabilities

## User namespaces

User Namespaces are a Linux-only feature to run workloads with increased isolation. How they work together with Pod Security Standards is described in the [documentation](https://kubernetes.io/docs/concepts/workloads/pods/user-namespaces/#integration-with-pod-security-admission-checks) for Pods that use user namespaces.

## FAQ

### Why isn't there a profile between Privileged and Baseline?

The three profiles defined here have a clear linear progression from most secure (Restricted) to least secure (Privileged), and cover a broad set of workloads. Privileges required above the Baseline policy are typically very application specific, so we do not offer a standard profile in this niche. This is not to say that the privileged profile should always be used in this case, but that policies in this space need to be defined on a case-by-case basis.

SIG Auth may reconsider this position in the future, should a clear need for other profiles arise.

### What's the difference between a security profile and a security context?

[Security Contexts](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/) configure Pods and Containers at runtime. Security contexts are defined as part of the Pod and container specifications in the Pod manifest, and represent parameters to the container runtime.

Security profiles are control plane mechanisms to enforce specific settings in the Security Context, as well as other related parameters outside the Security Context. As of July 2021, [Pod Security Policies](https://kubernetes.io/docs/concepts/security/pod-security-policy/) are deprecated in favor of the built-in [Pod Security Admission Controller](https://kubernetes.io/docs/concepts/security/pod-security-admission/).

### What about sandboxed Pods?

There is currently no API standard that controls whether a Pod is considered sandboxed or not. Sandbox Pods may be identified by the use of a sandboxed runtime (such as gVisor or Kata Containers), but there is no standard definition of what a sandboxed runtime is.

The protections necessary for sandboxed workloads can differ from others. For example, the need to restrict privileged permissions is lessened when the workload is isolated from the underlying kernel. This allows for workloads requiring heightened permissions to still be isolated.

Additionally, the protection of sandboxed workloads is highly dependent on the method of sandboxing. As such, no single recommended profile is recommended for all sandboxed workloads.

  

Last modified August 06, 2025 at 6:48 PM PST: [nit-fix: Add empty value for host field in probes PSA (a0fb9cc6b3)](https://github.com/kubernetes/website/commit/a0fb9cc6b3bdc96b6df50a6ab6778140150ea484)