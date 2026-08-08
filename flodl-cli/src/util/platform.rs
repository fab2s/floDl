//! Which OS family this box is, and the commands that mean the same
//! thing on each.
//!
//! Guidance that names a package or a service is only useful if it names
//! the right one, and the three families flodl targets disagree on all
//! of it: the ssh unit is `ssh` on Debian and `sshd` on RHEL, a
//! non-standard ssh port needs an SELinux label on RHEL and nothing on
//! Debian, Ubuntu hands the ssh listener to a socket unit that ignores
//! the `Port` directive, and macOS has no systemd at all.
//!
//! Everything here is a pure function of the family, so the whole table
//! is testable from any host. [`Platform::detect`] is the only impure
//! entry point.

/// Where a command is being suggested to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Debian, Ubuntu and derivatives: apt, `ssh.service`, ufw.
    Debian,
    /// RHEL, Rocky, Alma, CentOS, Fedora: dnf, `sshd.service`,
    /// firewalld, and SELinux in the way of a non-standard port.
    Rhel,
    /// macOS: no systemd, Remote Login instead of a package, and the
    /// ssh port is not an ordinary config edit.
    MacOs,
    /// Something else Unix-like. Guidance degrades to naming the goal
    /// rather than inventing a command.
    Other,
}

impl Platform {
    /// This host's family, from `/etc/os-release` on Linux.
    pub fn detect() -> Self {
        if cfg!(target_os = "macos") {
            return Platform::MacOs;
        }
        if !cfg!(target_os = "linux") {
            return Platform::Other;
        }
        match std::fs::read_to_string("/etc/os-release") {
            Ok(c) => Self::from_os_release(&c),
            Err(_) => Platform::Other,
        }
    }

    /// Pure parse of an os-release body: `ID` or any `ID_LIKE` token
    /// decides the family. Debian is the fallback on Linux because it
    /// is what the shipped images and cloud hosts are, so an unknown
    /// derivative gets the more likely of two wrong answers.
    pub fn from_os_release(body: &str) -> Self {
        const RHEL: &[&str] = &["rhel", "fedora", "centos", "rocky", "almalinux"];
        const DEB: &[&str] = &["debian", "ubuntu"];
        let tokens: Vec<String> = body
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                l.strip_prefix("ID=").or_else(|| l.strip_prefix("ID_LIKE="))
            })
            .flat_map(|v| {
                v.trim_matches('"')
                    .split_whitespace()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect();
        if tokens.iter().any(|t| RHEL.contains(&t.as_str())) {
            return Platform::Rhel;
        }
        if tokens.iter().any(|t| DEB.contains(&t.as_str())) {
            return Platform::Debian;
        }
        Platform::Debian
    }

    /// Install one or more packages, named in this family's spelling.
    /// `None` where there is no package manager to name.
    pub fn install(&self, packages: &[&str]) -> Option<String> {
        let list = packages.join(" ");
        match self {
            Platform::Debian => Some(format!("sudo apt install -y {list}")),
            Platform::Rhel => Some(format!("sudo dnf install -y {list}")),
            Platform::MacOs => Some(format!("brew install {list}")),
            Platform::Other => None,
        }
    }

    /// The package providing an ssh server.
    pub fn sshd_package(&self) -> Option<&'static str> {
        match self {
            Platform::Debian | Platform::Rhel => Some("openssh-server"),
            // Shipped; enabled through Remote Login instead.
            Platform::MacOs | Platform::Other => None,
        }
    }

    /// The systemd unit serving ssh. macOS has none.
    pub fn ssh_service(&self) -> Option<&'static str> {
        match self {
            Platform::Debian => Some("ssh.service"),
            Platform::Rhel => Some("sshd.service"),
            Platform::MacOs | Platform::Other => None,
        }
    }

    /// Bring the ssh daemon up on boot and now.
    ///
    /// On Debian the socket unit must be disabled first, and this is the
    /// step whose absence is most confusing: while `ssh.socket` owns the
    /// listener, the `Port` directive in `sshd_config` is IGNORED, so a
    /// carefully written drop-in appears to do nothing at all.
    pub fn enable_sshd(&self) -> Vec<String> {
        match self {
            Platform::Debian => vec![
                "sudo systemctl disable --now ssh.socket".to_string(),
                "sudo systemctl enable --now ssh.service".to_string(),
            ],
            Platform::Rhel => vec!["sudo systemctl enable --now sshd.service".to_string()],
            Platform::MacOs => vec!["sudo systemsetup -setremotelogin on".to_string()],
            Platform::Other => vec![],
        }
    }

    /// Open a TCP port on the host firewall, when this family has one
    /// that is on by default.
    pub fn open_port(&self, port: u16) -> Option<String> {
        match self {
            Platform::Debian => Some(format!("sudo ufw allow {port}/tcp   # if ufw is active")),
            Platform::Rhel => Some(format!(
                "sudo firewall-cmd --permanent --add-port={port}/tcp && sudo firewall-cmd --reload"
            )),
            Platform::MacOs | Platform::Other => None,
        }
    }

    /// Let sshd bind a non-standard port. Only SELinux systems need
    /// this, and without it the daemon fails to start with a permission
    /// error that says nothing about SELinux.
    pub fn allow_ssh_port(&self, port: u16) -> Option<String> {
        match self {
            Platform::Rhel if port != 22 => Some(format!(
                "sudo semanage port -a -t ssh_port_t -p tcp {port}   \
                 # SELinux; needs policycoreutils-python-utils"
            )),
            _ => None,
        }
    }

    /// Whether a drop-in under `/etc/ssh/sshd_config.d/` is read by
    /// default. macOS ships no `Include` line on older releases, and
    /// its ssh port is owned by launchd rather than the config anyway.
    pub fn has_sshd_config_d(&self) -> bool {
        matches!(self, Platform::Debian | Platform::Rhel)
    }

    /// How to get a RUNNABLE `rrsync` here — the forced command door
    /// `b` puts on its key.
    ///
    /// Debian ships it executable in the `rsync` package
    /// (`/usr/bin/rrsync`), so installing rsync is the whole answer.
    /// RHEL ships the same script as DOCUMENTATION:
    /// `/usr/share/doc/rsync/support/rrsync`, mode 0644 and a python3
    /// script, so it is neither on PATH nor executable. Installing rsync
    /// there is necessary and not sufficient, and a door composed with a
    /// bare `rrsync` fails with the least helpful error sshd has.
    pub fn rrsync_fix(&self) -> Option<String> {
        match self {
            Platform::Debian => self.install(&["rsync"]),
            Platform::Rhel => Some(
                concat!(
                    "sudo dnf install -y rsync && ",
                    "sudo install -m 755 /usr/share/doc/rsync/support/rrsync ",
                    "/usr/local/bin/rrsync",
                )
                .to_string(),
            ),
            // Homebrew's rsync ships the script under its own prefix and
            // the path moves with the platform, so name the goal.
            Platform::MacOs => Some(
                "brew install rsync, then put its support/rrsync on PATH as an executable"
                    .to_string(),
            ),
            Platform::Other => None,
        }
    }

    /// Whether this process is inside a container.
    ///
    /// It changes what a fix MEANS, not just where it runs: a package
    /// installed into a running container lives in a writable layer that
    /// the next `docker compose run --rm` throws away, so the durable
    /// answer is the image, and advice that omits that sends the
    /// operator round the same loop tomorrow.
    pub fn in_container() -> bool {
        if std::path::Path::new("/.dockerenv").exists() {
            return true;
        }
        std::fs::read_to_string("/proc/1/cgroup")
            .map(|c| c.contains("docker") || c.contains("containerd") || c.contains("libpod"))
            .unwrap_or(false)
    }

    /// A short human name for reports.
    pub fn name(&self) -> &'static str {
        match self {
            Platform::Debian => "Debian/Ubuntu",
            Platform::Rhel => "RHEL/Fedora",
            Platform::MacOs => "macOS",
            Platform::Other => "this OS",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn os_release_picks_the_family_from_id_or_id_like() {
        let rocky = "NAME=\"Rocky Linux\"\nID=\"rocky\"\nID_LIKE=\"rhel centos fedora\"\n";
        let fedora = "ID=fedora\n";
        let ubuntu = "NAME=\"Ubuntu\"\nID=ubuntu\nID_LIKE=debian\n";
        let debian = "ID=debian\n";
        assert_eq!(Platform::from_os_release(rocky), Platform::Rhel);
        assert_eq!(Platform::from_os_release(fedora), Platform::Rhel);
        assert_eq!(Platform::from_os_release(ubuntu), Platform::Debian);
        assert_eq!(Platform::from_os_release(debian), Platform::Debian);
        // Unknown derivative: Debian is the likelier of two wrong
        // answers, since that is what the images and cloud hosts are.
        assert_eq!(Platform::from_os_release("ID=weird\n"), Platform::Debian);
    }

    #[test]
    fn the_ssh_unit_and_socket_trap_differ_by_family() {
        // The Debian pair is the load-bearing one: while ssh.socket owns
        // the listener, `Port` in sshd_config is ignored outright.
        let deb = Platform::Debian.enable_sshd();
        assert!(
            deb.iter().any(|c| c.contains("disable --now ssh.socket")),
            "{deb:?}"
        );
        assert!(
            deb.iter().any(|c| c.contains("enable --now ssh.service")),
            "{deb:?}"
        );
        assert_eq!(Platform::Debian.ssh_service(), Some("ssh.service"));

        let rhel = Platform::Rhel.enable_sshd();
        assert!(rhel.iter().any(|c| c.contains("sshd.service")), "{rhel:?}");
        assert!(
            !rhel.iter().any(|c| c.contains("socket")),
            "no socket unit on RHEL: {rhel:?}"
        );
        assert_eq!(Platform::Rhel.ssh_service(), Some("sshd.service"));
    }

    #[test]
    fn selinux_labeling_is_named_only_where_it_bites() {
        // A non-standard port on RHEL fails to bind without the label,
        // with an error that never mentions SELinux.
        assert!(Platform::Rhel.allow_ssh_port(2022).is_some());
        assert!(Platform::Rhel.allow_ssh_port(22).is_none());
        assert!(Platform::Debian.allow_ssh_port(2022).is_none());
        assert!(Platform::MacOs.allow_ssh_port(2022).is_none());
    }

    #[test]
    fn container_detection_answers_without_panicking() {
        // Both answers are legitimate depending on where the suite runs
        // (this repo's own tests run inside the dev image); what matters
        // is that the probe is total.
        let _ = Platform::in_container();
    }

    #[test]
    fn rrsync_guidance_knows_rhel_ships_it_unexecutable() {
        // Debian's rsync package puts an executable rrsync on PATH, so
        // installing rsync is the whole answer there.
        let deb = Platform::Debian.rrsync_fix().unwrap();
        assert!(deb.contains("apt") && deb.contains("rsync"), "{deb}");
        assert!(
            !deb.contains("install -m"),
            "no copy needed on Debian: {deb}"
        );
        // RHEL ships it as docs: 0644, not on PATH. Installing rsync is
        // necessary and NOT sufficient, so the fix must also place it.
        let rhel = Platform::Rhel.rrsync_fix().unwrap();
        assert!(
            rhel.contains("install -m 755"),
            "must make it executable: {rhel}"
        );
        assert!(
            rhel.contains("/usr/share/doc/rsync/support/rrsync"),
            "{rhel}"
        );
    }

    #[test]
    fn package_commands_speak_each_families_manager() {
        assert!(
            Platform::Debian
                .install(&["rsync"])
                .unwrap()
                .starts_with("sudo apt")
        );
        assert!(
            Platform::Rhel
                .install(&["rsync"])
                .unwrap()
                .starts_with("sudo dnf")
        );
        assert!(
            Platform::MacOs
                .install(&["rsync"])
                .unwrap()
                .starts_with("brew")
        );
        assert!(Platform::Other.install(&["rsync"]).is_none());
        // macOS ships sshd, so there is no package to name for it.
        assert_eq!(Platform::MacOs.sshd_package(), None);
        assert_eq!(Platform::Debian.sshd_package(), Some("openssh-server"));
    }
}
