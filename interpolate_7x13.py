"""
Interpolate 7x13E propeller performance from 7x11E and 7x15E data.
Linear interpolation at the midpoint (pitch 13 between 11 and 15).
"""
import re

def parse_file(filepath):
    """Parse a PER3 performance file into structured RPM sections."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    sections = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r'\s+PROP RPM\s*=\s*(\d+)', line)
        if match:
            rpm = int(match.group(1))
            # Skip header lines (RPM line, blank, column headers, units)
            i += 1  # blank or header
            # Find the column header line
            while i < len(lines) and 'V ' not in lines[i]:
                i += 1
            i += 1  # skip column header
            i += 1  # skip units line

            # Read data rows
            data_rows = []
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped == '':
                    i += 1
                    break
                # Parse numeric values
                vals = stripped.split()
                if len(vals) >= 15:
                    data_rows.append([float(v) for v in vals])
                i += 1

            sections.append((rpm, data_rows))
        else:
            i += 1

    return sections


def interpolate_sections(sec11, sec15, weight=0.5):
    """Interpolate between two sets of RPM sections."""
    # Build lookup by RPM
    dict11 = {rpm: rows for rpm, rows in sec11}
    dict15 = {rpm: rows for rpm, rows in sec15}

    # Use RPMs common to both
    common_rpms = sorted(set(dict11.keys()) & set(dict15.keys()))

    result = []
    for rpm in common_rpms:
        rows11 = dict11[rpm]
        rows15 = dict15[rpm]

        n_rows = min(len(rows11), len(rows15))
        interp_rows = []
        for j in range(n_rows):
            row = []
            for k in range(len(rows11[j])):
                v = rows11[j][k] * (1 - weight) + rows15[j][k] * weight
                row.append(v)
            interp_rows.append(row)

        result.append((rpm, interp_rows))

    return result


def format_output(sections):
    """Format interpolated data into the PER3 file format."""
    lines = []
    lines.append("         7x13E                    (7x13E.dat)")
    lines.append("         Interpolated from 7x11E and 7x15E")
    lines.append("         Estimation Date: 04/16/2026")
    lines.append("")
    lines.append("         AIRFOIL AERO DATA GENERATED USING:  INTERPOLATION (7x11E + 7x15E)  ")
    lines.append("")
    lines.append("         ====== PERFORMANCE DATA (versus advance ratio and MPH) ======")
    lines.append("")
    lines.append("         DEFINITIONS:")
    lines.append("         J=V/nD (advance ratio) ")
    lines.append("         Ct=T/(rho * n**2 * D**4) (thrust coef.)")
    lines.append("         Cp=P/(rho * n**3 * D**5) (power coef.)")
    lines.append("         Pe=Ct*J/Cp (efficiency) ")
    lines.append("         V  (model speed in MPH) ")
    lines.append("         Mach (at prop tip)  ")
    lines.append("         Reyn (at 75% of span) ")
    lines.append("         FOM (Figure of Merit)")
    lines.append("")
    lines.append("")

    for rpm, rows in sections:
        lines.append(f"         PROP RPM = {rpm:10d}")
        lines.append("")
        lines.append("         V          J           Pe         Ct          Cp          PWR         Torque      Thrust      PWR         Torque      Thrust      THR/PWR      Mach      Reyn       FOM")
        lines.append("       (mph)     (Adv_Ratio)     -          -           -          (Hp)        (In-Lbf)     (Lbf)      (W)         (N-m)       (N)         (g/W)         -         -          -")

        for row in rows:
            # Format: V, J, Pe, Ct, Cp, PWR(Hp), Torque(In-Lbf), Thrust(Lbf), PWR(W), Torque(N-m), Thrust(N), THR/PWR, Mach, Reyn, FOM
            # Columns: 0=V, 1=J, 2=Pe, 3=Ct, 4=Cp, 5=PWR_hp, 6=Torque_inlbf, 7=Thrust_lbf,
            #          8=PWR_W, 9=Torque_Nm, 10=Thrust_N, 11=THR/PWR, 12=Mach, 13=Reyn, 14=FOM
            line = (f"  {row[0]:10.2f}  {row[1]:10.4f}  {row[2]:10.4f}  {row[3]:10.4f}  {row[4]:10.4f}  "
                    f"{row[5]:10.3f}  {row[6]:10.3f}  {row[7]:10.3f}  "
                    f"{row[8]:10.3f}  {row[9]:10.3f}  {row[10]:10.3f}  "
                    f"{row[11]:10.3f}    {row[12]:10.2f}  {row[13]:8.0f}.  {row[14]:10.4f}")
            lines.append(line)

        lines.append("")
        lines.append("")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    sec11 = parse_file("PER3_7x11E.txt")
    sec15 = parse_file("PER3_7x15E.txt")

    print(f"7x11E: {len(sec11)} RPM sections, RPMs: {[s[0] for s in sec11]}")
    print(f"7x15E: {len(sec15)} RPM sections, RPMs: {[s[0] for s in sec15]}")

    interp = interpolate_sections(sec11, sec15, weight=0.5)
    print(f"7x13E: {len(interp)} RPM sections (common RPMs)")

    output = format_output(interp)

    with open("PER3_7x13E.txt", "w") as f:
        f.write(output)

    print("Written PER3_7x13E.txt")
